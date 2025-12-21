from pydantic import BaseModel

class InputText(BaseModel):
    text: str
    model: str
