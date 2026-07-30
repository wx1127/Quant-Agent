"""Request IDs and uniform API error responses."""

import logging
import re
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from quant_agent.core.errors import ErrorCode, QuantAgentError
from quant_agent.observability.logging import log_context
from quant_agent.observability.redaction import redact

_logger = logging.getLogger("quant_agent.api")

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")

_STATUS = {
    ErrorCode.INVALID_ARGUMENT: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.DATA_UNAVAILABLE: 503,
    ErrorCode.DATA_INVALID: 422,
    ErrorCode.RISK_REJECTED: 409,
    ErrorCode.APPROVAL_REQUIRED: 403,
    ErrorCode.APPROVAL_EXPIRED: 410,
    ErrorCode.KILL_SWITCH_ACTIVE: 423,
    ErrorCode.INTERNAL_ERROR: 500,
}


def install_error_handling(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied = request.headers.get("X-Request-ID")
        request_id = supplied if supplied and _SAFE_REQUEST_ID.fullmatch(supplied) else None
        request.state.request_id = request_id or f"req_{uuid4().hex}"
        with log_context(request_id=request.state.request_id):
            try:
                response = await call_next(request)
            except Exception:
                response = JSONResponse(
                    status_code=500,
                    content=_error_payload(
                        request,
                        ErrorCode.INTERNAL_ERROR,
                        "internal service error",
                        {},
                    ),
                )
            _logger.info(
                "api request completed method=%s path=%s status=%s",
                request.method,
                request.url.path,
                response.status_code,
                extra={"request_id": request.state.request_id},
            )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(QuantAgentError)
    async def quant_agent_error(request: Request, exc: QuantAgentError) -> JSONResponse:
        return JSONResponse(
            status_code=_STATUS[exc.code],
            content=_error_payload(request, exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request,
                ErrorCode.INVALID_ARGUMENT,
                "request validation failed",
                {
                    "errors": [
                        {
                            "type": item["type"],
                            "loc": item["loc"],
                            "msg": item["msg"],
                        }
                        for item in exc.errors()
                    ]
                },
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request,
                ErrorCode.INTERNAL_ERROR,
                "internal service error",
                {},
            ),
        )


def _error_payload(
    request: Request,
    code: ErrorCode,
    message: str,
    details: dict[str, object],
) -> dict[str, object]:
    return {
        "request_id": getattr(request.state, "request_id", None),
        "error": {
            "code": code.value,
            "message": message,
            "details": redact(details),
        },
    }
