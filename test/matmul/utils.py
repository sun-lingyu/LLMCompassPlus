from software_model.utils import DataType, data_type_dict
from test.utils import test_model_dict

def get_model_shape(model):
    model_shapes = test_model_dict[model]
    K_shapes = {
        "qkv_proj": model_shapes["hidden_size"], 
        "o_proj": model_shapes["head_dim"] * model_shapes["num_attention_heads"],
        "up_proj": model_shapes["hidden_size"],
        "down_proj": model_shapes["intermediate_size"]
        }
    assert(model_shapes["hidden_act"] in ["silu", "gelu"])
    N_shapes = {
        "qkv_proj": model_shapes["head_dim"] * (model_shapes["num_key_value_heads"] * 2 + model_shapes["num_attention_heads"]), 
        "o_proj": model_shapes["hidden_size"],
        "up_proj": model_shapes["intermediate_size"] * 2 if model_shapes["hidden_act"] == "silu" else model_shapes["intermediate_size"], # SiLU/GELU
        "down_proj": model_shapes["hidden_size"]
        }
    return K_shapes, N_shapes

def get_output_dtype(activation_dtype: DataType, op_name: str, is_test: bool):
    assert op_name in ("qkv_proj", "o_proj", "up_proj", "down_proj"), "op_name must be one of (qkv_proj, o_proj, up_proj, down_proj)"
    if activation_dtype.name in ("fp16", "int4"):
        return activation_dtype
    
    if activation_dtype.name == "int8":
        if is_test or op_name == "up_proj":
            return data_type_dict["int8"] # CUTLASS profiler does not support S8 GEMM with F16 output
        else:
            return data_type_dict["fp16"]
    
    if activation_dtype.name == "fp8":
        if op_name in ("qkv_proj", "up_proj"):
            return data_type_dict["fp8"]
        else:
            return data_type_dict["fp16"]
    
    if activation_dtype.name == "fp4":
        if op_name == "qkv_proj":
            return data_type_dict["fp8"]
        elif op_name == "up_proj":
            return data_type_dict["fp4"]
        else:
            if is_test:
                return data_type_dict["fp8"] # CUTLASS profiler does not support F4 GEMM with F16 output
            return data_type_dict["fp16"]
