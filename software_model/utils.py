from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        self.word_size:int = word_size

data_type_dict = {"int4": DataType("int4", 0.5), "int8": DataType("int8", 1), "int32": DataType("int32", 4), "fp4": DataType("fp4", 0.5), "fp8": DataType("fp8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}

class Tensor:
    def __init__(
        self, shape: List, data_type
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type
        
