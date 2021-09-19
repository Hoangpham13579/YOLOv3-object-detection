# YOLO-v3 model definition
You only look once, or YOLO, is one of the faster object detection algorithms out there. Though it is no longer the most accurate object detection algorithm, it is a very good choice when you need real-time detection, without loss of too much accuracy. In this case, we'll discover Yolo-v3 model construction

<img src="https://www.researchgate.net/profile/Paolo-Valdez/publication/341369179/figure/fig1/AS:890935600750593@1589427008836/The-YOLO-v3-architectuhttps://miro.medium.com/max/2000/1*d4Eg17IVJ0L41e7CTWLLSg.pngre.ppm" alt="drawing" width="800"/>

## 1. Feature extraction: Darknet-53
- Darknet-53 network is used for performing feature extraction. The network uses successive 3x3 and 1x1 convolutional layers but it includes some shortcut connections as well and is significantly large (53 convolutional layers)
- Darknet-53 also achieves the highest measured floating point operations per second. Better utilize GPU, more efficient to evaluate and this faster.
- File: `Darknet53_Feature_Extraction.py`

<img src="https://www.researchgate.net/publication/338121987/figure/fig4/AS:839282637938698@1577111982954/Structure-of-the-Darknet53-convolutional-network.png" alt="drawing" width="600"/>

## 2. Convolution layers
- Yolo has a large number of convolutional layers in the middle of construction. 
- File: `Yolo_Convolution_Layer.py`

## 3. Detection layers
The most salient feature of YOLO v3 is that it makes detection at three different scales, which is implemented by downsampling the dimensions of the input image by 32, 16 and 8 respectively
- For each cell in the feature map, the detection layers predicts `n_anchors * (5 + n_classes)` values using 1x1 convolution
- For each scale we have `n_anchors = 3`
- File: `Detection_Layer.py`

<img src="https://miro.medium.com/max/626/1*Dx_VzLT94QoZAAyWbmC57A.png" alt="drawing" width="400"/>
Reference: YOLOv3 paper

## 4. Upsample layer
- In order to concatenate with shortcut outputs from Darknet-53 before applying detection layers, we are going to upsample the feature map using **nearest neighbor interpolation**. (The yellow part in Yolov3 construction)
- File: `Upsample_Layer.py`

## 5. Non-max suppression
- The model is going to draw (detect) a lot of boxes for only 1 object, the author chooses the box with highest confidence probability & compute IOU (Intersection of Unit) to decide whether a box is kept or discarded. This technique is called "Non-max Suppression"
- File: `Non_Max_Suppression.py`

## 6. Final model YOLOv3
- Finally, with all the ready recipes, file `Yolo_v3.py` implements the constructions of YOLOv3 model from features extraction using Darknet53 to detection layer.
- File: `Yolo_v3.py`

## More resources to explore Yolo-v3 model: 
- Blog: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
- Paper: [YOLOv3 paper](https://arxiv.org/pdf/1804.02767.pdf)
