from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException, Depends
from starlette.status import HTTP_403_FORBIDDEN
from config import Settings, get_settings
from fastapi import Header, HTTPException
from pydantic import BaseModel
from starlette import status

api_key_header = APIKeyHeader(name="access_token", auto_error=False)


async def get_api_key(settings: Settings = Depends(get_settings), api_key_header: str = Security(api_key_header)):
    print(settings)
    print(api_key_header)
    if api_key_header == settings.API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
        )




# Placeholder for a database containing valid token values
known_tokens = set(["api_token_1234token"])


class UnauthorizedMessage(BaseModel):
    detail: str = "Bearer token missing or unknown"


import typing as t
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


# We will handle a missing token ourselves
get_bearer_token = HTTPBearer(auto_error=False)


async def get_token(
    auth: t.Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    settings = get_settings()
    known_tokens = set([settings.API_BEARER_TOKEN])
    # Simulate a database query to find a known token
    if auth is None or (token := auth.credentials) not in known_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )
    return token