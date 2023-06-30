import argparse
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from logging import getLogger
from libtiff import TIFF
import numpy as np
from json import dump

logger = getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model prediction on an image")
    parser.add_argument("--input_data_file", type=str, help="Input file path",
                        default='/data/purple_box/'
                                'FKP4_L57D855P1_topro_purplebox_x200y1400z0530.tif')
    parser.add_argument("--output_data_file", type=str, help="Output file path",
                        default='/data/purple_box/FKP4_L57D855P1_topro_purplebox_x200y1400z0530.json')
    parser.add_argument("--model_file", type=str, default='/data/model/sam_vit_h_4b8939.pth',
                        help="path for the model to load for inference")

    args = parser.parse_args()
    input_data_file = args.input_data_file
    output_data_file = args.output_data_file
    model_file = args.model_file

    # load model
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=model_file)
    sam.to(device='cuda')

    # load input tiff image into numpy array in RGB format
    tif = TIFF.open(input_data_file)
    images = []
    for image in tif.iter_images():
        images.append(image)

    # generate masks for the entire image
    mask_generator = SamAutomaticMaskGenerator(sam)
    output = {}
    slice_no = 0
    for im in images:
        # cast image down from 16 bits to 8 bits
        image = im.astype("uint8")
        # convert image to RGB by replicating each intensity into RGB components added as the third axis
        image_3d = image[:, :, np.newaxis].repeat(3, axis=2)
        # masks is a list of dicts containing all segmented masks for the image slice
        masks = mask_generator.generate(image_3d)
        output[slice_no] = masks
        slice_no += 1

    with open(output_data_file, 'w') as fp:
        dump(output, fp)
