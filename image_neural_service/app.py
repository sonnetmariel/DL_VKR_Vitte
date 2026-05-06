# app.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from config import (
    CONFIG,
    AppConfig,
    get_artifact_status,
    get_config,
    get_public_config_summary,
    validate_required_artifacts,
)
from image_utils import (
    ImageValidationError,
    path_to_url,
    save_uploaded_file,
)
from llm_client import generate_llm_report, get_llm_client, mask_secrets_in_text
from model_registry import ModelRegistryError, get_model_registry
from report_service import ReportServiceError, get_report_service
from restoration_service import (
    RestorationServiceError,
    get_available_degradation_types,
    get_available_restoration_modes,
    get_restoration_service,
)
from segmentation_service import (
    SegmentationServiceError,
    get_available_background_variants,
    get_available_segmentation_modes,
    get_segmentation_service,
)
from storage import (
    StorageError,
    get_history_stats,
    get_storage_service,
    list_history,
)


def create_app(config: AppConfig | None = None) -> Flask:
    """
    Создает Flask-приложение.

    Приложение объединяет два экспериментальных результата:
    сегментацию объекта животного и восстановление качества изображения.
    """
    app_config = config or get_config()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = app_config.secret_key
    app.config["MAX_CONTENT_LENGTH"] = app_config.runtime.max_content_length_bytes

    register_template_context(app, app_config)
    register_routes(app, app_config)
    register_error_handlers(app)

    return app


def register_template_context(app: Flask, config: AppConfig) -> None:
    """
    Добавляет общие переменные во все HTML-шаблоны.
    """

    @app.context_processor
    def inject_common_context() -> dict[str, Any]:
        artifact_problems = validate_required_artifacts(config)

        return {
            "app_name": config.app_name,
            "app_version": config.app_version,
            "debug_mode": config.debug,
            "artifact_problems": artifact_problems,
            "llm_enabled": config.llm.use_llm_explanation,
            "allowed_extensions": ", ".join(config.runtime.allowed_extensions),
            "max_content_length_mb": config.runtime.max_content_length_mb,
        }


def register_routes(app: Flask, config: AppConfig) -> None:
    """
    Регистрирует маршруты приложения.
    """

    @app.route("/", methods=["GET"])
    def index():
        artifact_problems = validate_required_artifacts(config)

        return render_template(
            "index.html",
            page_title="Загрузка изображения",
            operation_types=get_operation_types(),
            segmentation_modes=get_available_segmentation_modes(),
            restoration_modes=get_available_restoration_modes(),
            degradation_types=get_available_degradation_types(),
            background_variants=get_available_background_variants(),
            artifact_problems=artifact_problems,
            default_segmentation_mode=config.segmentation.default_mode,
            default_restoration_mode=config.restoration.default_mode,
            default_degradation_type=config.restoration.default_degradation_type,
        )

    @app.route("/process", methods=["POST"])
    def process_image():
        artifact_problems = validate_required_artifacts(config)

        if artifact_problems:
            for problem in artifact_problems:
                flash(problem, "error")

            return redirect(url_for("index"))

        if "image" not in request.files:
            flash("Файл изображения не был передан.", "error")
            return redirect(url_for("index"))

        uploaded_file = request.files["image"]

        if not uploaded_file or not uploaded_file.filename:
            flash("Файл изображения не выбран.", "error")
            return redirect(url_for("index"))

        operation_type = normalize_operation_type(
            request.form.get("operation_type", "full_pipeline")
        )

        source_filename = uploaded_file.filename or "uploaded_image.png"
        uploaded_info = None

        try:
            uploaded_info = save_uploaded_file(
                file_storage=uploaded_file,
                upload_folder=config.runtime.upload_folder,
                config=config,
            )

            use_llm = checkbox_enabled(request.form.get("use_llm"))
            llm_force = checkbox_enabled(request.form.get("force_llm"))

            if operation_type == "segmentation":
                record = run_segmentation_request(
                    config=config,
                    input_path=Path(uploaded_info.absolute_path),
                    source_filename=source_filename,
                    use_llm=use_llm,
                    force_llm=llm_force,
                )

            elif operation_type == "restoration":
                record = run_restoration_request(
                    config=config,
                    input_path=Path(uploaded_info.absolute_path),
                    source_filename=source_filename,
                    use_llm=use_llm,
                    force_llm=llm_force,
                )

            elif operation_type == "full_pipeline":
                record = run_full_pipeline_request(
                    config=config,
                    input_path=Path(uploaded_info.absolute_path),
                    source_filename=source_filename,
                    use_llm=use_llm,
                    force_llm=llm_force,
                )

            else:
                raise ValueError(f"Неизвестный сценарий обработки: {operation_type}")

            flash("Обработка изображения завершена.", "success")
            return redirect(url_for("result", record_uid=record.record_uid))

        except Exception as exc:
            safe_error = mask_secrets_in_text(str(exc))
            error_trace = mask_secrets_in_text(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )

            input_path = ""
            input_url = ""

            if uploaded_info is not None:
                input_path = str(uploaded_info.absolute_path)
                input_url = path_to_url(Path(uploaded_info.absolute_path), config)

            try:
                storage = get_storage_service()
                record = storage.save_error_record(
                    operation_type=operation_type,
                    source_filename=source_filename,
                    error_text=safe_error,
                    input_path=input_path,
                    input_url=input_url,
                    extra={
                        "traceback": error_trace,
                    },
                )
                flash("Во время обработки возникла ошибка. Запись сохранена в истории.", "error")
                return redirect(url_for("result", record_uid=record.record_uid))

            except Exception:
                flash(f"Во время обработки возникла ошибка: {safe_error}", "error")
                return redirect(url_for("index"))

    @app.route("/result/<record_uid>", methods=["GET"])
    def result(record_uid: str):
        storage = get_storage_service()
        record = storage.get_record_by_uid(record_uid)

        if record is None:
            abort(404)

        display_report = extract_display_report(record.data)
        llm_result = record.data.get("llm_result") if isinstance(record.data, dict) else None

        return render_template(
            "result.html",
            page_title="Результат обработки",
            record=record,
            display_report=display_report,
            llm_result=llm_result,
            result_data=record.data,
        )

    @app.route("/history", methods=["GET"])
    def history():
        operation_type = request.args.get("operation_type") or None
        status = request.args.get("status") or None

        page = parse_positive_int(request.args.get("page"), default=1)
        per_page = parse_positive_int(request.args.get("per_page"), default=20)

        if per_page > 100:
            per_page = 100

        offset = (page - 1) * per_page

        records = list_history(
            limit=per_page,
            offset=offset,
            operation_type=operation_type,
            status=status,
        )

        stats = get_history_stats()
        total_count = get_storage_service().count_records(
            operation_type=operation_type,
            status=status,
        )

        has_next = offset + per_page < total_count
        has_prev = page > 1

        return render_template(
            "history.html",
            page_title="История обработки",
            records=records,
            stats=stats,
            page=page,
            per_page=per_page,
            total_count=total_count,
            has_next=has_next,
            has_prev=has_prev,
            operation_type=operation_type,
            status=status,
            operation_types=get_operation_types(),
        )

    @app.route("/history/delete/<int:record_id>", methods=["POST"])
    def delete_history_record(record_id: int):
        storage = get_storage_service()

        deleted = storage.delete_record(record_id)

        if deleted:
            flash("Запись истории удалена.", "success")
        else:
            flash("Запись истории не найдена.", "error")

        return redirect(url_for("history"))

    @app.route("/history/clear", methods=["POST"])
    def clear_history_route():
        storage = get_storage_service()
        deleted_count = storage.clear_history()

        flash(f"История очищена. Удалено записей: {deleted_count}.", "success")

        return redirect(url_for("history"))

    @app.route("/about", methods=["GET"])
    def about():
        report_service = get_report_service()
        about_data = report_service.build_about_report_data()

        model_status = get_model_registry().status()

        return render_template(
            "about.html",
            page_title="О системе",
            about_data=about_data,
            model_status=model_status,
            config_summary=get_public_config_summary(config),
            artifact_status=get_artifact_status(config),
            llm_diagnostics=get_llm_client().diagnostics(),
        )

    @app.route("/runtime/<section>/<path:filename>", methods=["GET"])
    def runtime_file(section: str, filename: str):
        folder_map = {
            "uploads": config.runtime.upload_folder,
            "results": config.runtime.result_folder,
            "previews": config.runtime.preview_folder,
            "reports": config.runtime.report_folder,
        }

        folder = folder_map.get(section)

        if folder is None:
            abort(404)

        return send_from_directory(str(folder), filename)

    @app.route("/api/status", methods=["GET"])
    def api_status():
        return jsonify(
            {
                "app": get_public_config_summary(config),
                "artifact_status": get_artifact_status(config),
                "artifact_problems": validate_required_artifacts(config),
                "history_stats": get_history_stats().to_dict(),
                "llm": get_llm_client().diagnostics(),
            }
        )

    @app.route("/api/warmup", methods=["POST"])
    def api_warmup():
        result = get_model_registry().warmup()

        return jsonify(result)

    @app.route("/api/history", methods=["GET"])
    def api_history():
        limit = parse_positive_int(request.args.get("limit"), default=20)
        offset = parse_positive_int(request.args.get("offset"), default=0)

        if limit > 100:
            limit = 100

        records = [
            record.to_dict()
            for record in list_history(limit=limit, offset=offset)
        ]

        return jsonify(
            {
                "records": records,
                "stats": get_history_stats().to_dict(),
            }
        )


def run_segmentation_request(
    config: AppConfig,
    input_path: Path,
    source_filename: str,
    use_llm: bool,
    force_llm: bool,
):
    """
    Выполняет сценарий сегментации и сохраняет запись истории.
    """
    segmentation_mode = request.form.get(
        "segmentation_mode",
        config.segmentation.default_mode,
    )
    background_variant = request.form.get("background_variant", "soft_blue")
    blur_radius = parse_positive_int(request.form.get("blur_radius"), default=18)

    segmentation_service = get_segmentation_service()
    report_service = get_report_service()
    storage = get_storage_service()

    segmentation_result = segmentation_service.process_image(
        input_path=input_path,
        mode=segmentation_mode,
        background_variant=background_variant,
        blur_radius=blur_radius,
        postprocess_mask=True,
    )

    report = report_service.build_segmentation_report(
        segmentation_result=segmentation_result,
        save=True,
        source_filename=source_filename,
    )

    llm_result = None

    if use_llm or force_llm:
        llm_result = generate_llm_report(
            payload=report.payload_for_llm,
            fallback_markdown=report.markdown_text,
            force=force_llm or use_llm,
        )

    return storage.save_processing_record(
        operation_type="segmentation",
        source_filename=source_filename,
        result=segmentation_result,
        report=report,
        llm_result=llm_result,
        status="success",
        input_path=str(input_path),
        input_url=path_to_url(input_path, config),
    )


def run_restoration_request(
    config: AppConfig,
    input_path: Path,
    source_filename: str,
    use_llm: bool,
    force_llm: bool,
):
    """
    Выполняет сценарий восстановления качества и сохраняет запись истории.
    """
    restoration_mode = request.form.get(
        "restoration_mode",
        config.restoration.default_mode,
    )
    degradation_type = request.form.get(
        "degradation_type",
        config.restoration.default_degradation_type,
    )

    restoration_service = get_restoration_service()
    report_service = get_report_service()
    storage = get_storage_service()

    restoration_result = restoration_service.process_image(
        input_path=input_path,
        mode=restoration_mode,
        degradation_type=degradation_type,
    )

    report = report_service.build_restoration_report(
        restoration_result=restoration_result,
        save=True,
        source_filename=source_filename,
    )

    llm_result = None

    if use_llm or force_llm:
        llm_result = generate_llm_report(
            payload=report.payload_for_llm,
            fallback_markdown=report.markdown_text,
            force=force_llm or use_llm,
        )

    return storage.save_processing_record(
        operation_type="restoration",
        source_filename=source_filename,
        result=restoration_result,
        report=report,
        llm_result=llm_result,
        status="success",
        input_path=str(input_path),
        input_url=path_to_url(input_path, config),
    )


def run_full_pipeline_request(
    config: AppConfig,
    input_path: Path,
    source_filename: str,
    use_llm: bool,
    force_llm: bool,
):
    """
    Выполняет полный конвейер.

    Сначала применяется восстановление качества, затем результат восстановления
    передается в сервис сегментации и обработки фона.
    """
    restoration_mode = request.form.get(
        "restoration_mode",
        config.restoration.default_mode,
    )
    degradation_type = request.form.get(
        "degradation_type",
        config.restoration.default_degradation_type,
    )
    segmentation_mode = request.form.get(
        "segmentation_mode",
        config.segmentation.default_mode,
    )
    background_variant = request.form.get("background_variant", "soft_blue")
    blur_radius = parse_positive_int(request.form.get("blur_radius"), default=18)

    restoration_service = get_restoration_service()
    segmentation_service = get_segmentation_service()
    report_service = get_report_service()
    storage = get_storage_service()

    restoration_result = restoration_service.process_image(
        input_path=input_path,
        mode=restoration_mode,
        degradation_type=degradation_type,
    )

    restored_image_path = Path(restoration_result.result_image.absolute_path)

    segmentation_result = segmentation_service.process_image(
        input_path=restored_image_path,
        mode=segmentation_mode,
        background_variant=background_variant,
        blur_radius=blur_radius,
        postprocess_mask=True,
    )

    report = report_service.build_full_pipeline_report(
        restoration_result=restoration_result,
        segmentation_result=segmentation_result,
        save=True,
        source_filename=source_filename,
    )

    pipeline_result = {
        "input_path": str(input_path),
        "input_url": path_to_url(input_path, config),
        "restoration_result": restoration_result.to_dict(),
        "segmentation_result": segmentation_result.to_dict(),
    }

    llm_result = None

    if use_llm or force_llm:
        llm_result = generate_llm_report(
            payload=report.payload_for_llm,
            fallback_markdown=report.markdown_text,
            force=force_llm or use_llm,
        )

    return storage.save_processing_record(
        operation_type="full_pipeline",
        source_filename=source_filename,
        result=pipeline_result,
        report=report,
        llm_result=llm_result,
        status="success",
        input_path=str(input_path),
        input_url=path_to_url(input_path, config),
    )


def register_error_handlers(app: Flask) -> None:
    """
    Регистрирует обработчики ошибок.

    Важно: Flask не принимает кортеж классов в декораторе app.errorhandler.
    Поэтому несколько пользовательских исключений регистрируются через цикл.
    """

    @app.errorhandler(400)
    def bad_request(error):
        return render_template(
            "result.html",
            page_title="Ошибка запроса",
            record=None,
            display_report="Некорректный запрос.",
            llm_result=None,
            result_data={"error": str(error)},
        ), 400

    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "result.html",
            page_title="Страница не найдена",
            record=None,
            display_report="Запрошенная страница или запись истории не найдена.",
            llm_result=None,
            result_data={"error": str(error)},
        ), 404

    @app.errorhandler(413)
    def file_too_large(error):
        config = get_config()
        flash(
            f"Файл слишком большой. Максимальный размер: "
            f"{config.runtime.max_content_length_mb} МБ.",
            "error",
        )
        return redirect(url_for("index"))

    @app.errorhandler(ImageValidationError)
    def image_validation_error(error):
        flash(str(error), "error")
        return redirect(url_for("index"))

    def service_error(error):
        safe_error = mask_secrets_in_text(str(error))

        return render_template(
            "result.html",
            page_title="Ошибка обработки",
            record=None,
            display_report=f"Во время обработки возникла ошибка: {safe_error}",
            llm_result=None,
            result_data={"error": safe_error},
        ), 500

    service_exception_classes = (
        ModelRegistryError,
        SegmentationServiceError,
        RestorationServiceError,
        ReportServiceError,
        StorageError,
    )

    for exception_class in service_exception_classes:
        app.register_error_handler(exception_class, service_error)

    @app.errorhandler(Exception)
    def unexpected_error(error):
        safe_error = mask_secrets_in_text(str(error))

        if get_config().debug:
            safe_trace = mask_secrets_in_text(
                "".join(traceback.format_exception(type(error), error, error.__traceback__))
            )
        else:
            safe_trace = ""

        return render_template(
            "result.html",
            page_title="Непредвиденная ошибка",
            record=None,
            display_report=(
                "Во время работы приложения возникла непредвиденная ошибка. "
                f"{safe_error}"
            ),
            llm_result=None,
            result_data={
                "error": safe_error,
                "traceback": safe_trace,
            },
        ), 500


def get_operation_types() -> list[dict[str, str]]:
    """
    Возвращает сценарии обработки для формы на главной странице.
    """
    return [
        {
            "value": "full_pipeline",
            "name_ru": "полный конвейер: восстановление качества и обработка фона",
        },
        {
            "value": "restoration",
            "name_ru": "только восстановление качества",
        },
        {
            "value": "segmentation",
            "name_ru": "только сегментация и обработка фона",
        },
    ]


def normalize_operation_type(value: str | None) -> str:
    """
    Приводит тип операции к внутреннему названию.
    """
    if not value:
        return "full_pipeline"

    normalized = value.strip().lower()

    aliases = {
        "pipeline": "full_pipeline",
        "full": "full_pipeline",
        "full_pipeline": "full_pipeline",
        "полный конвейер": "full_pipeline",
        "restoration": "restoration",
        "restore": "restoration",
        "восстановление": "restoration",
        "segmentation": "segmentation",
        "segment": "segmentation",
        "сегментация": "segmentation",
    }

    return aliases.get(normalized, "full_pipeline")


def checkbox_enabled(value: Any) -> bool:
    """
    Проверяет значение флажка HTML-формы.
    """
    if value is None:
        return False

    return str(value).strip().lower() in {"1", "true", "yes", "on", "да"}


def parse_positive_int(value: Any, default: int) -> int:
    """
    Безопасно читает положительное целое число из формы.
    """
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        return default

    if result < 0:
        return default

    return result


def extract_display_report(record_data: dict[str, Any]) -> str:
    """
    Выбирает текст отчета для страницы результата.

    Если языковая модель успешно сформировала отчет, показывается ее текст.
    Иначе показывается локальный отчет, созданный приложением.
    """
    if not isinstance(record_data, dict):
        return ""

    llm_result = record_data.get("llm_result")

    if isinstance(llm_result, dict):
        if llm_result.get("success") and llm_result.get("text"):
            return str(llm_result["text"])

    report = record_data.get("report")

    if isinstance(report, dict):
        markdown_text = report.get("markdown_text")

        if markdown_text:
            return str(markdown_text)

    error_text = record_data.get("error")

    if error_text:
        return str(error_text)

    return "Отчет для записи истории отсутствует."


app = create_app(CONFIG)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=CONFIG.debug,
    )
