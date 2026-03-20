from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RpcCommand:
    type: str
    id: str | None = None
    payload: dict[str, Any] | None = None
