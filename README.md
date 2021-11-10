# Yolo-v3 object detection 
This code is implemented based on the concept of paper Yolo-v3 (You Only Look Once_version 3). Yolo-v3 is a real-time object detection algorithm that identifies specific objects in videos, images or in realtime. It is known to be an incredibly performance, state-of-the-art model architecture: fast, accurate, and reliable. This model is trained on [COCO dataset](https://cocodataset.org/#home) to detect 80 different objects totally and it requires huge amount of computations for training. Therefore, I'll utilize [pre-trained model Yolo-v3](https://pjreddie.com/darknet/yolo/) trained by YOLOv3's paper author with the help of Opencv library to detect objects on image, on video and on the lap's camera

## To-do list
- [x] Yolo-v3 architecture
- [x] Yolo-v3 object detection on Image
- [x] Yolo-v3 object detection on Video
- [x] Yolo-v3 object detection on Camera (Real-time)

## Code requirement
`pip install requirements.txt` to install needed libraries

## File orgization
```Shell
├── YOLOV3-object-detection (Current directory)
    ├── Model_architecture
        ├── Darknet53_Feature_Extraction.py
        └── Detection_layer.py
        └── Non_Max_Suppression.py 
        └── Upsample_Layer.py
        └── Yolo_Convolution_Layer.py
        └── Yolo_v3.py
        └── README.md : (Explanation of the above model architecture)
    ├── data : (Data used to detect objects)
        ├── vung_tau_old.jpg
        ├── yolo_test.mp4
    ├── models : (Yolo-v3 models)
        ├── coco.names
        └── yolov3.cfg
        └── yolov3.weights
    ├── YOLOv3_camera.py : (Run object detection on our own computer)
    ├── YOLOv3_image.py : (Run object detection on image)
    ├── YOLOv3_video.py : (Run object detection on video)
    ├── requirements.txt : (Requirement libraries)
```

## How to Use
If you want to understand more about the architecture of model including their implementations by using tensorflow, explore folder `Model_architecture` 
- Command below is used to apply YOLOv3 model to detect objects on the image. You can add an image to folder "data", and modify the path to apply on your own image

`python YOLOv3_image.py --image_path ./data/vung_tau.jpg`
- Command below is used to apply YOLOv3 model to detect objects on a video. Use the same instruction above to apply your own video (Push "q" to quit)

`python YOLOv3_video.py --video_path ./data/yolo_test.mp4`
- Command below is used to apply YOLOv3 model to detect objects on your camera (Push "q" to quit)

`python YOLOv3_camera.py`

## Result
![](https://github.com/HarryPham0123/YOLOv3-object-detection/blob/main/data/YOLOv3_result.gif)

## Reference/Credits (Special thanks to)
- YOLOv3: An Incremental Improvement (2018) _ [Paper](https://arxiv.org/pdf/1804.02767.pdf)
- Deep learning Specialization on Coursera _ taught by Andrew Ng [link](https://www.coursera.org/specializations/deep-learning)
- Pre-trained model trained by YOLOv3's paper author _ https://pjreddie.com/darknet/yolo/
- The implementation was also inspired a lot by [Yolov3 github of Paweł Kapica](https://github.com/mystic123/tensorflow-yolo-v3)
