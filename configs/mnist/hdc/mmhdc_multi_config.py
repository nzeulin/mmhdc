from configs.mnist.hdc.default_hdc_config import get_default_config

def get_config():
    config = get_default_config()
    config.name = 'mmhdc_multi'
    config.C = 500

    return config