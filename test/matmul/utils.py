from software_model.utils import DataType, data_type_dict

test_model_dict = {
    "InternVision":{
        "head_dim": 64,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "hidden_act": "gelu"
    },
    "Qwen3_0_6B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8 ,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "hidden_act": "silu"
    },
    "Qwen3_1_7B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8 ,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "hidden_act": "silu"
    },
    "Qwen3_4B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8 ,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "hidden_act": "silu"
    },
    "Qwen3_8B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8 ,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "hidden_act": "silu"
    }
}

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
            return data_type_dict["fp16"]
