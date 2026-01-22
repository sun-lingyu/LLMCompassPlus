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
