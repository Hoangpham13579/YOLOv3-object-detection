# Yolo-v3 object detection 
- This code is implemented based on the concept of paper Yolo-v3 (You Only Look Once_version 3). Yolo-v3 is a real-time object detection algorithm that identifies specific objects in videos, images or in realtime. It is known to be an incredibly performance, state-of-the-art model architecture: fast, accurate, and reliable.

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
- If you want to understand more about the architecture of model including their implementations by using tensorflow, explore folder `Model_architecture` 
