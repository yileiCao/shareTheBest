import torch
import os
import numpy as np
import cv2
from PIL import Image

from lut.lut_model import *
import torchvision.transforms.functional as TF


def retouch_lut_from_file(path_to_model_state, path_to_old_images, path_to_new_images):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    lut0, lut1, lut2, classifier, trilinear_ = load_model(path_to_model_state, cuda)
    items = os.listdir(path_to_old_images)
    for item in items:
        if item.endswith(".jpg"):
            load_path = os.path.join(path_to_old_images, item)
            save_path = os.path.join(path_to_new_images, item)
            image = Image.open(load_path)
            image = TF.to_tensor(image).type(Tensor)
            image = image.unsqueeze(0)
            lut = generate_LUT(image, lut0, lut1, lut2, classifier)
            _, result = trilinear_(lut, image)
            ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(save_path, quality=95)
            print(f"image complete:{item}")
    return 1


def retouch_lut(images, path_to_model_state="../../model_parameter"):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    lut0, lut1, lut2, classifier, trilinear_ = load_model(path_to_model_state, cuda)
    ims = []
    for image in images:
        image = TF.to_tensor(image).type(Tensor)
        image = image.unsqueeze(0)
        lut = generate_LUT(image, lut0, lut1, lut2, classifier)
        _, result = trilinear_(lut, image)
        image = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ims.append(image)
    return ims



def load_model(path_to_model_state, cuda):
    LUT0 = Generator3DLUT_zero()
    LUT1 = Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()
    # LUT3 = Generator3DLUT_zero()
    # LUT4 = Generator3DLUT_zero()
    classifier = Classifier()
    trilinear_ = TrilinearInterpolation()

    if cuda:
        LUT0 = LUT0.cuda()
        LUT1 = LUT1.cuda()
        LUT2 = LUT2.cuda()
        # LUT3 = LUT3.cuda()
        # LUT4 = LUT4.cuda()
        classifier = classifier.cuda()

    # Load pretrained models
    LUTs = torch.load(os.path.join(path_to_model_state, "LUTs.pth"), map_location=torch.device('cpu'))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # LUT3.load_state_dict(LUTs["3"])
    # LUT4.load_state_dict(LUTs["4"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    # LUT3.eval()
    # LUT4.eval()
    classifier.load_state_dict(torch.load(
        os.path.join(path_to_model_state, "classifier.pth"), map_location=torch.device('cpu')))
    classifier.eval()
    return LUT0, LUT1, LUT2, classifier, trilinear_


def generate_LUT(img, LUT0, LUT1, LUT2, classifier):
    pred = classifier(img).squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    return LUT


def main():
    retouch_lut_from_file("../model_parameter",
                "/image_folder",
                "save_folder")


if __name__ == "__main__":
    main()
