# Share The Best: Fast Photo Selection and Enhancement System
By Yilei Cao, Yixing Gao, Nora Horanyi, and Hyung Jin Chang

<img src="/img/IMAGE1-1.png" width="500" />


## Abstract
Automatic photo selection has been an actively studied area in computer vision research during the last decade. This task is very laborious for humans, especially when the photo album is very large. No-reference image aesthetic quality assessment (IAQA) networks are designed to solve this task by automatically assessing a given image based on aesthetic metrics. Despite good performance, most existing methods rely on deep networks and require plenty of computing resources when dealing with a large number of photos. In this work, we combined a state-of-the-art IAQA network and an image retouching network to build a fast image selection and enhancement system. To reduce the computing resources, we light-weighted the IAQA network and improved its pre- processing method. In addition, by exploring the relationship between IAQA networks and image retouching networks, we built the image retouching network with fast running speed and good enhancement performance. This image selection and enhancement system uses limited computing resources and runs fast on a non-GPU device.

## Contribution
1. We present a fast image selection and enhancement system that can process a large number of photos with limited computing resources.</li>
2. We trained and compared image aesthetic quality as- sessment networks based on different lightweight CNN models. We presented an image preprocessing method that is suitable for images with arbitrary aspect ratio.</li>
3. We explored the relationship between IAQA networks and image retouching networks. We proposed a new method to evaluate the performance of an image re- touching network using IAQA networks.

## Framework
![System Framework](/img/IMAGE2-1.png)


## Demo Usage
### Requirements
Python3, Requirements.txt

### Build
#### 3D-LUT trilinear package ([More details](https://github.com/HuiZeng/Image-Adaptive-3DLUT#image-adaptive-3dlut))
```bash
cd trilinear_cpp
sh setup.sh
```
### Run
```bash
cd PyQt_demo
python3 logic.py
```

## IAQA Training
After downloading [AVA benchmark](https://github.com/mtobeiyf/ava_downloader), NIMA-train.ipynb can be run to train IAQA models based on [PyTorch NIMA](https://github.com/truskovskiyk/nima.pytorch).

Trained model parameters can be found in model_parameter/para_iaqa

## Image Enhancement
MIT-Adobe-fiveK dataset can be downloaded with image_retouching/download_5k.py after download txt file from [MIT-5K](https://data.csail.mit.edu/graphics/fivek/)

image_retouching/lut/lut_eval.py is to enhance image folder with 3D-LUT model

image_retouching/csrnet/csr_eval.py is to enhance image folder with CSRNet model

## Acknowledgements
- [PyTorch NIMA](https://github.com/truskovskiyk/nima.pytorch)
- [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT#image-adaptive-3dlut)
- [CSRNet](https://github.com/hejingwenhejingwen/CSRNet)
