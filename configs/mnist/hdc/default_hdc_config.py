import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()

    # Training configuration
    config.name = ''
    config.learning_rate = 1e-5
    config.normalize = True
    config.zero_bias = True

    return config