import yaml


class SAMConfig:
    models = {}
    data = {}

    def __init__(self):
        pass


def init_configuration(config_file_path):
    """
    Initialize SAMConfig Singleton instance from configuration file
    """
    with open(config_file_path) as config_stream:
        config = yaml.load(config_stream, Loader=yaml.SafeLoader)
        SAMConfig.models = config['model']
        SAMConfig.data = config['data']
