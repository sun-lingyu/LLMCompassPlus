from software_model.utils import DataType, data_type_dict


def get_output_dtype(model_dtype: DataType, is_test: bool):
    """Fused SiLU/Mul output; Thor FP8 uses static E4M3 quantization."""
    if model_dtype.name == "fp8":
        return data_type_dict["fp8"]
    if model_dtype.name == "int4":
        return data_type_dict["fp16"]
    return model_dtype
