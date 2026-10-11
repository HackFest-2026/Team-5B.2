"""Consistent API errors: every failure is ``{"error": {"code", "message", "details"}}``."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.engine.context import InvalidInputError


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def not_found(what: str, ident: str) -> ApiError:
    return ApiError(404, "NOT_FOUND", f'{what} "{ident}" not found')


def _body(code: str, message: str, details: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return JSONResponse(_body(e.code, e.message, e.details), status_code=e.status)

    @app.exception_handler(InvalidInputError)
    async def _input(_: Request, e: InvalidInputError):
        return JSONResponse(_body("INVALID_INPUT", "The request has invalid attribute values.", e.errors), status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [
            {"path": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
            for err in e.errors()
        ]
        return JSONResponse(_body("VALIDATION_ERROR", "The request body is invalid.", details), status_code=422)
