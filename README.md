# shareTheBest: Fast Photo Selection and Enhancement System
<h2>Abstract</h2>
<p>Automatic photo selection has been an actively studied area in computer vision research during the last decade. This task is very laborious for humans, especially when the photo album is very large. No-reference image aesthetic quality assessment (IAQA) networks are designed to solve this task by automatically assessing a given image based on aesthetic metrics. Despite good performance, most existing methods rely on deep networks and require plenty of computing resources when dealing with a large number of photos. In this work, we combined a state-of-the-art IAQA network and an image retouching network to build a fast image selection and enhancement system. To reduce the computing resources, we light-weighted the IAQA network and improved its pre- processing method. In addition, by exploring the relationship between IAQA networks and image retouching networks, we built the image retouching network with fast running speed and good enhancement performance. This image selection and enhancement system uses limited computing resources and runs fast on a non-GPU device.</p>

<h2>Contribution</h2>
<ol>
  <li>We present a fast image selection and enhancement system that can process a large number of photos with limited computing resources.</li>
  <li>We trained and compared image aesthetic quality as- sessment networks based on different lightweight CNN models. We presented an image preprocessing method that is suitable for images with arbitrary aspect ratio.</li>
  <li>We explored the relationship between IAQA networks and image retouching networks. We proposed a new method to evaluate the performance of an image re- touching network using IAQA networks.</li>
</ol>

<h2>Framework</h2>

<h2>Demo Usage</h2>
<h3>Requirements</h3>
<p>Python3, requirements.txt</p>

<h3>Build</h3>

<h3>Run</h3>

<h2>IAQA Training(In progress)</h2>

<h2>Image Enhancement(In progress)</h2>

<h2>Acknowledgements</h2>
[PyTorch NIMA](https://github.com/truskovskiyk/nima.pytorch)
[Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT#image-adaptive-3dlut)
[CSRNet](https://github.com/hejingwenhejingwen/CSRNet)
