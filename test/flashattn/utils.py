from software_model.utils import DataType, data_type_dict
from test.utils import test_model_dict


def get_model_shape(model):
    model_shapes = test_model_dict[model]
    head_dim = model_shapes["head_dim"]
    num_heads_q = model_shapes["num_attention_heads"]
    num_heads_kv = model_shapes["num_key_value_heads"]
    return head_dim, num_heads_q, num_heads_kv


def get_output_dtype(model_dtype: DataType, is_test: bool):
    assert model_dtype in ("fp16", "int8", "int4", "fp8", "fp4")
    if is_test:
        if model_dtype in ("fp16", "int8", "int4"):
            return data_type_dict["fp16"]
        else:
            return data_type_dict["fp8"]
    else:
        if model_dtype.name == "int4":
            return data_type_dict["fp16"]
        else:
            return model_dtype
