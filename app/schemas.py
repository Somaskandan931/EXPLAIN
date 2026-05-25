from pydantic import BaseModel
from typing import Optional


class InputText(BaseModel):
    text: str
    model: Optional[str] = None   # None → auto-detect language and select model