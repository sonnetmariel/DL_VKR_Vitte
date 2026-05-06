# llm_client.py
from __future__ import annotations

import json
import re
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from config import AppConfig, get_config


class LLMClientError(RuntimeError):
    """
    Базовая ошибка обращения к языковой модели.
    """


class LLMConfigError(LLMClientError):
    """
    Ошибка настроек языковой модели.
    """


class LLMAuthError(LLMClientError):
    """
    Ошибка ключа доступа или прав доступа.
    """


class LLMRateLimitError(LLMClientError):
    """
    Ошибка ограничения количества запросов или баланса.
    """


class LLMHTTPError(LLMClientError):
    """
    Ошибка ответа программного интерфейса.
    """

    def __init__(self, status_code: int, response_body: str) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"Ошибка HTTP {status_code}: {response_body[:500]}")


@dataclass(frozen=True)
class LLMCallResult:
    """
    Результат обращения к языковой модели.
    """

    success: bool
    used_model: str | None
    text: str
    source: str
    error_message: str | None
    elapsed_seconds: float
    created_at: str
    raw_response: dict[str, Any] | None
    request_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mask_secret(value: str) -> str:
    """
    Маскирует ключ доступа для журналов и диагностических сообщений.
    """
    if not value:
        return "не указан"

    if len(value) <= 8:
        return "***"

    return value[:4] + "***" + value[-4:]


def mask_secrets_in_text(text: str) -> str:
    """
    Удаляет ключи доступа и токены из текста ошибки.
    """
    if not text:
        return text

    patterns = [
        r"sk-[A-Za-z0-9_\-]{8,}",
        r"Bearer\s+[A-Za-z0-9_\-\.]+",
        r"api[_-]?key['\"]?\s*[:=]\s*['\"]?[^,'\"\s]+",
        r"token['\"]?\s*[:=]\s*['\"]?[^,'\"\s]+",
    ]

    masked = text

    for pattern in patterns:
        masked = re.sub(pattern, "***", masked, flags=re.IGNORECASE)

    return masked


def build_chat_completions_url(base_url: str) -> str:
    """
    Формирует адрес метода chat completions.
    """
    base_url = (base_url or "").strip().rstrip("/")

    if not base_url:
        raise LLMConfigError("Не указан адрес программного интерфейса языковой модели.")

    if base_url.endswith("/chat/completions"):
        return base_url

    return base_url + "/chat/completions"


def split_model_names(model_name: str) -> list[str]:
    """
    Возвращает список моделей.

    Если в настройке указано несколько моделей через запятую, они будут проверены
    по очереди. Обычно достаточно одной экономичной модели.
    """
    values = [
        item.strip()
        for item in str(model_name or "").split(",")
        if item.strip()
    ]

    if not values:
        return ["gpt-4o-mini"]

    return values


def compact_json(data: dict[str, Any], max_chars: int = 8000) -> str:
    """
    Сериализует данные в компактный текст для запроса.

    Ограничение защищает от случайной отправки слишком большого отчета.
    """
    text = json.dumps(data, ensure_ascii=False, indent=2)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n... Данные были сокращены из-за ограничения длины запроса."


def build_system_prompt() -> str:
    """
    Системная инструкция для языковой модели.
    """
    return (
        "Ты формируешь краткий пользовательский отчет по результатам нейросетевой "
        "обработки изображения. Пиши на русском языке, спокойно и понятно. "
        "Не выдумывай чисел, файлов и выводов. Используй только переданные данные. "
        "Если показатель является справочным результатом эксперимента, явно называй его "
        "справочным, а не индивидуальной оценкой загруженного изображения. "
        "Не утверждай, что один режим лучше по всем показателям, если в данных есть "
        "различия между метриками. Укажи ограничения результата и практическую рекомендацию."
    )


def build_user_prompt(payload: dict[str, Any]) -> str:
    """
    Пользовательская часть запроса.
    """
    return (
        "Сформируй отчет по данным ниже. Структура отчета: назначение обработки, "
        "выполненный режим, основные наблюдения, ограничения, рекомендация. "
        "Объем не больше 1800 знаков.\n\n"
        "Данные обработки:\n"
        f"{compact_json(payload)}"
    )


def make_error_message(exc: Exception) -> str:
    """
    Делает техническую ошибку понятной для интерфейса.
    """
    text = str(exc)

    if isinstance(exc, LLMAuthError):
        return (
            "Не удалось обратиться к языковой модели: ключ доступа не принят "
            "или у ключа нет прав на выбранную модель."
        )

    if isinstance(exc, LLMRateLimitError):
        return (
            "Не удалось обратиться к языковой модели: достигнуто ограничение запросов "
            "или закончился доступный баланс."
        )

    if isinstance(exc, LLMConfigError):
        return f"Не удалось обратиться к языковой модели: {text}"

    if isinstance(exc, LLMHTTPError):
        if exc.status_code == 401:
            return "Не удалось обратиться к языковой модели: неверный ключ доступа."
        if exc.status_code == 403:
            return "Не удалось обратиться к языковой модели: доступ запрещен."
        if exc.status_code == 404:
            return "Не удалось обратиться к языковой модели: модель или адрес не найдены."
        if exc.status_code == 429:
            return "Не удалось обратиться к языковой модели: превышено ограничение запросов."
        return f"Не удалось обратиться к языковой модели: ошибка HTTP {exc.status_code}."

    if "timed out" in text.lower() or "timeout" in text.lower():
        return "Не удалось обратиться к языковой модели: превышено время ожидания."

    return "Не удалось обратиться к языковой модели. Использован локальный отчет."


class LLMClient:
    """
    Клиент для обращения к ChatGPT через ProxyAPI-совместимый интерфейс.

    Клиент не отправляет изображения. Во внешний сервис передаются только численные
    показатели, режимы обработки и предупреждения, подготовленные report_service.py.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    @property
    def enabled(self) -> bool:
        return bool(self.config.llm.use_llm_explanation)

    def validate_settings(self) -> None:
        """
        Проверяет наличие обязательных настроек.
        """
        if not self.config.llm.proxyapi_api_key:
            raise LLMConfigError("не указан PROXYAPI_API_KEY в файле .env")

        if not self.config.llm.openai_base_url:
            raise LLMConfigError("не указан OPENAI_BASE_URL в файле .env")

        if not self.config.llm.model_name:
            raise LLMConfigError("не указана MODEL_NAME в файле .env")

    def build_request_body(
        self,
        model_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Формирует тело запроса к методу chat completions.
        """
        return {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(payload),
                },
            ],
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
        }

    def call_model(
        self,
        model_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Выполняет один запрос к выбранной модели.
        """
        self.validate_settings()

        url = build_chat_completions_url(self.config.llm.openai_base_url)
        body = self.build_request_body(model_name, payload)

        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.llm.proxyapi_api_key}",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.llm.timeout_seconds,
            ) as response:
                response_text = response.read().decode("utf-8")

            return json.loads(response_text)

        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8")
            except Exception:
                response_body = ""

            response_body = mask_secrets_in_text(response_body)

            if exc.code in {401, 403}:
                raise LLMAuthError(response_body) from exc

            if exc.code == 429:
                raise LLMRateLimitError(response_body) from exc

            raise LLMHTTPError(exc.code, response_body) from exc

        except urllib.error.URLError as exc:
            raise LLMClientError(f"Сетевая ошибка: {exc}") from exc

        except json.JSONDecodeError as exc:
            raise LLMClientError("Ответ языковой модели не является корректным JSON.") from exc

    def extract_text(self, response: dict[str, Any]) -> str:
        """
        Извлекает текст ответа из результата chat completions.
        """
        choices = response.get("choices", [])

        if not choices:
            raise LLMClientError("В ответе языковой модели нет вариантов ответа.")

        first_choice = choices[0]

        message = first_choice.get("message", {})

        if isinstance(message, dict):
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

        text = first_choice.get("text")

        if isinstance(text, str) and text.strip():
            return text.strip()

        raise LLMClientError("В ответе языковой модели нет текста.")

    def generate_report_text(
        self,
        payload: dict[str, Any],
        fallback_markdown: str = "",
        force: bool = False,
    ) -> LLMCallResult:
        """
        Создает пользовательский отчет через языковую модель.

        force=True используется для кнопки в интерфейсе. Если force=False и в .env
        отключен USE_LLM_EXPLANATION, возвращается локальный отчет без внешнего запроса.
        """
        start_time = time.time()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        request_summary = {
            "base_url": self.config.llm.openai_base_url,
            "models": split_model_names(self.config.llm.model_name),
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "timeout_seconds": self.config.llm.timeout_seconds,
            "api_key": mask_secret(self.config.llm.proxyapi_api_key),
            "force": force,
            "enabled_in_env": self.enabled,
        }

        if not force and not self.enabled:
            return LLMCallResult(
                success=False,
                used_model=None,
                text=fallback_markdown,
                source="local_report",
                error_message=(
                    "Обращение к языковой модели отключено в настройках. "
                    "Использован локальный отчет."
                ),
                elapsed_seconds=round(time.time() - start_time, 4),
                created_at=created_at,
                raw_response=None,
                request_summary=request_summary,
            )

        last_error: Exception | None = None

        for model_name in split_model_names(self.config.llm.model_name):
            try:
                response = self.call_model(
                    model_name=model_name,
                    payload=payload,
                )

                text = self.extract_text(response)

                return LLMCallResult(
                    success=True,
                    used_model=model_name,
                    text=text,
                    source="llm",
                    error_message=None,
                    elapsed_seconds=round(time.time() - start_time, 4),
                    created_at=created_at,
                    raw_response=response,
                    request_summary=request_summary,
                )

            except Exception as exc:
                last_error = exc
                continue

        error_message = make_error_message(last_error) if last_error else "Неизвестная ошибка."

        raw_error = ""
        if last_error is not None:
            raw_error = "".join(
                traceback.format_exception(
                    type(last_error),
                    last_error,
                    last_error.__traceback__,
                )
            )
            raw_error = mask_secrets_in_text(raw_error)

        return LLMCallResult(
            success=False,
            used_model=None,
            text=fallback_markdown,
            source="local_report",
            error_message=error_message + (f"\n\nТехнические сведения:\n{raw_error}" if raw_error else ""),
            elapsed_seconds=round(time.time() - start_time, 4),
            created_at=created_at,
            raw_response=None,
            request_summary=request_summary,
        )

    def generate_short_recommendation(
        self,
        payload: dict[str, Any],
        fallback_text: str,
        force: bool = False,
    ) -> LLMCallResult:
        """
        Формирует короткую рекомендацию по результату обработки.

        Метод можно использовать отдельно от полного отчета, например для блока
        на странице результата.
        """
        compact_payload = {
            "task": "short_recommendation",
            "instruction": (
                "Сформируй одну короткую рекомендацию для пользователя. "
                "Не больше трех предложений. Не выдумывай чисел."
            ),
            "data": payload,
        }

        return self.generate_report_text(
            payload=compact_payload,
            fallback_markdown=fallback_text,
            force=force,
        )

    def diagnostics(self) -> dict[str, Any]:
        """
        Возвращает безопасную диагностику настроек языковой модели.
        """
        return {
            "enabled": self.enabled,
            "base_url": self.config.llm.openai_base_url,
            "models": split_model_names(self.config.llm.model_name),
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "timeout_seconds": self.config.llm.timeout_seconds,
            "api_key": mask_secret(self.config.llm.proxyapi_api_key),
        }


def get_llm_client() -> LLMClient:
    """
    Создает клиент языковой модели.
    """
    return LLMClient()


def generate_llm_report(
    payload: dict[str, Any],
    fallback_markdown: str = "",
    force: bool = False,
) -> LLMCallResult:
    """
    Удобная функция для вызова языковой модели из app.py.
    """
    client = get_llm_client()

    return client.generate_report_text(
        payload=payload,
        fallback_markdown=fallback_markdown,
        force=force,
    )
