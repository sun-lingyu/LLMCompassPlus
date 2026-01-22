from software_model.utils import DataType, data_type_dict
from test.utils import test_model_dict

def get_model_shape(model):
    model_shapes = test_model_dict[model]
    N_shape = test_model_dict[model]["hidden_size"]
    return N_shape