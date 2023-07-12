import yaml
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class SAMModel:
    model_name = ''
    model_predictor = None

    def __init__(self):
        pass

    @staticmethod
    def set_uploaded_image(image_file):
        img = Image.open(image_file).convert("RGB")
        SAMModel.model_predictor.set_image(np.array(img))


def init_model(config_file_path):
        """
        Initialize SAMModel instance from configuration file
        """
        with open(config_file_path) as config_stream:
            config = yaml.load(config_stream, Loader=yaml.SafeLoader)
            model_type = config['model']['type']
            SAMModel.model_name = config['model']['name']
            model_path = config['model']['path']
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device='cuda')
            SAMModel.model_predictor = SamPredictor(sam)
