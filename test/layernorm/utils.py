from test.utils import test_model_dict


def get_model_shape(model):
    N_shape = test_model_dict[model]["hidden_size"]
    return N_shape
