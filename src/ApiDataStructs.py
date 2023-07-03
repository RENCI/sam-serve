from pydantic import BaseModel


class ModelResponse(BaseModel):
    name: str


class DataInfoResponse(BaseModel):
    name: str
    total_slices: int
