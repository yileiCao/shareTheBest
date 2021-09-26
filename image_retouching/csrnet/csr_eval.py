import torch
import os
import numpy as np
import cv2
from PIL import Image
from csr_model import csr_network
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def csr_retouch(path_to_model_state, path_to_old_images, path_to_new_images):
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    network = csr_network()

    network.load_state_dict(torch.load(
        path_to_model_state, map_location=torch.device('cpu')))
    network.eval()
    # img = image_file_to_tensor(image_path)


    # result = network(img)
    items = os.listdir(path_to_old_images)
    for item in items:
        if item.endswith(".jpg"):
            load_path = os.path.join(path_to_old_images, item)
            save_path = os.path.join(path_to_new_images, item)
            image = Image.open(load_path)
            image = TF.to_tensor(image).type(Tensor)
            image = image.unsqueeze(0)
            result = network(image)
            result = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(result)
            im.save(save_path, quality=95)
    return 1

'''
def image_file_to_tensor(image_path):
    items = os.listdir(image_path)
    img = Image.open(os.path.join(image_path, items[0])).convert("RGB")
    width, height = img.size
    # images = torch.zeros(len(items), 3, height, width)
    images = torch.zeros(1, 3, height, width, requires_grad=False)
    index = 0
    for item in items:
        if item.endswith(".jpg"):
            load_path = os.path.join(image_path, item)
            image = Image.open(load_path).convert("RGB")
            image = TF.to_tensor(image).type(torch.FloatTensor)
            images[index, :, :, :] = image
            index += 1
        if index >= 1:
            break
    return images
'''

def main():
    csr_retouch("../../model_parameter/csrnet.pth", ".../image_folder", ".../save_folder)


if __name__ == "__main__":
    main()
