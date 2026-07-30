"""Bearer authentication and explicit role permissions."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from quant_agent.core.errors import ErrorCode, QuantAgentError


class Role(StrEnum):
    VIEWER = "VIEWER"
    RESEARCHER = "RESEARCHER"
    TRADER = "TRADER"
    APPROVER = "APPROVER"
    RISK_ADMIN = "RISK_ADMIN"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    roles: frozenset[Role]
    account_ids: frozenset[str] = frozenset()


class AuthService:
    """Stores only token hashes; raw bearer tokens remain outside the service."""

    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}

    def register_token(self, token: str, principal: Principal) -> None:
        if len(token) < 16:
            raise ValueError("API token must be at least 16 characters")
        digest = _digest(token)
        if digest in self._principals:
            raise ValueError("API token is already registered")
        self._principals[digest] = principal

    def authenticate(self, token: str) -> Principal:
        principal = self._principals.get(_digest(token))
        if principal is None:
            raise QuantAgentError(ErrorCode.UNAUTHORIZED, "invalid or expired credentials")
        return principal


_bearer = HTTPBearer(auto_error=False)


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise QuantAgentError(ErrorCode.UNAUTHORIZED, "bearer authentication is required")
    auth: AuthService = request.app.state.auth
    return auth.authenticate(credentials.credentials)


def require_roles(*allowed: Role) -> Callable[..., Principal]:
    allowed_set = frozenset(allowed)

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.roles & allowed_set:
            raise QuantAgentError(ErrorCode.FORBIDDEN, "role is not permitted for this operation")
        return principal

    return dependency


def require_account(principal: Principal, account_id: str) -> None:
    if account_id not in principal.account_ids and Role.SYSTEM_ADMIN not in principal.roles:
        raise QuantAgentError(ErrorCode.FORBIDDEN, "account is outside the caller scope")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
