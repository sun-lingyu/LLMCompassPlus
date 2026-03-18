from software_model.utils import DataType, data_type_dict
from test.utils import test_model_dict


def get_model_shape(model):
    N_shape = test_model_dict[model]["hidden_size"]
    return N_shape


def get_output_dtype(model_dtype: DataType, is_test: bool):
    if is_test:
        return data_type_dict["fp16"]
    else:
        if model_dtype.name == "int4":
            return data_type_dict["fp16"]
        else:
            return model_dtype
