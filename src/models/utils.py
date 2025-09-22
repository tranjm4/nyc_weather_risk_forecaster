from typing_extensions import TypedDict

class ModelResponse(TypedDict):
    mean: float
    std: float
    lower_95: float
    upper_95: float