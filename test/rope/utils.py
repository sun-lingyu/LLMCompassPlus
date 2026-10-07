from software_model.utils import DataType, data_type_dict
from test.utils import test_model_dict


def get_model_shape(model):
    head_dim = test_model_dict[model]["head_dim"]
    num_heads_q = test_model_dict[model]["num_attention_heads"]
    num_heads_kv = test_model_dict[model]["num_key_value_heads"]
    return head_dim, num_heads_q, num_heads_kv


def get_output_dtype(model_dtype: DataType, is_test: bool):
    if model_dtype.name == "fp8":  # Thor FP8 execution contract
        return data_type_dict["fp16"]
    if is_test:
        return data_type_dict["fp16"]
    else:
        if model_dtype.name == "int4":
            return data_type_dict["fp16"]
        else:
            return model_dtype
