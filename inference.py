
import numpy as np
import sys
from util import ImageProcessing

import torch
import torchvision.transforms.functional as TF
import util

import model

np.set_printoptions(threshold=sys.maxsize)


checkpoint_filepath = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
inference_img_dirpath = "./adobe5k_dpe"
log_dirpath = './'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

net = model.CURLNet()
checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
net.to(DEVICE)


def load_image(path):
    '''
    '''

    input_img = util.ImageProcessing.load_image(path, normaliser = 1)
    input_img = input_img.astype(np.uint8)
                
    return TF.to_tensor(TF.to_pil_image(input_img)).to(DEVICE)


def evaluate(img_path):
    """
    """
    img = load_image(img_path)

    with torch.no_grad():

        img = img.unsqueeze(0)
        img = torch.clamp(img, 0, 1)

        net_output_img_example , _ = net(img)

        net_output_img_example_numpy = net_output_img_example.squeeze(0).data.cpu().numpy()
        net_output_img_example_numpy = ImageProcessing.swapimdims_3HW_HW3(net_output_img_example_numpy)
        net_output_img_example_rgb = net_output_img_example_numpy
        net_output_img_example_rgb = ImageProcessing.swapimdims_HW3_3HW(net_output_img_example_rgb)
        net_output_img_example_rgb = np.expand_dims(net_output_img_example_rgb, axis=0)
        net_output_img_example_rgb = np.clip(net_output_img_example_rgb, 0, 1)

        net_output_img_example = (net_output_img_example_rgb[0, 0:3, :, :] * 255).astype('uint8')
        return ImageProcessing.swapimdims_3HW_HW3(net_output_img_example)
