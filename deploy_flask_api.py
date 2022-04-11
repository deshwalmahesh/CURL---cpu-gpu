'''
the app requires 4 parameters in POST request:
img: Image as a Stream object
name: Name of the image
size: Resize the image. My 4GB CUDA goes out of memory. Dynamic for now, remove it in production
'''

import torch
import torchvision.transforms.functional as TF
import numpy as np

import util
from util import ImageProcessing, Image
import model

from io import BytesIO
from flask import Flask, request, jsonify
import base64


app = Flask(__name__)  # Flask App

checkpoint_filepath = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
inference_img_dirpath = "./adobe5k_dpe"
log_dirpath = './'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

net = model.CURLNet()
checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
net.to(DEVICE)


def load_image(path, size):
    '''
    Load image
    '''
    input_img = util.ImageProcessing.load_image(path, normaliser = 1, size = size)
    input_img = input_img.astype(np.uint8)
                
    return TF.to_tensor(TF.to_pil_image(input_img)).to(DEVICE)


@app.route('/enhance', methods = ['POST'])
def enhance():
    """
    Infer annd return
    """
    try:
        image_data = request.files["img"]
        name = request.form['name']
        ext = name.split('.')[-1]
        size = int(request.form['size'])
        img = load_image(BytesIO(image_data.read()), size = size)

    except Exception as e:
        return jsonify([f"Image Reading error: {e}"]),400

    try:
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
            img = ImageProcessing.swapimdims_3HW_HW3(net_output_img_example)

            img = Image.fromarray(img)
            buffer = BytesIO()
            img.save(buffer, ext) # get extension with request
            buffer.seek(0)
            data = buffer.read()
            data = base64.b64encode(data).decode()
            return jsonify({'img': data, "name":name.replace('.','_enhanced.')})
            
    except Exception as e:
         return jsonify([f"Inference error: {e}"]),400


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0')

