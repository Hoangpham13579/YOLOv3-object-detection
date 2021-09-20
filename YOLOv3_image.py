import cv2
import numpy as np


# we are not going to bother with objects less than 30% probability
THRESHOLD = 0.3
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.3
YOLO_IMAGE_SIZE = 320


def find_objects(model_outputs):
    """
        Extract the the values from prediction vectors resulted by the YOLOv3 algorithm
        Returns:
            box_indexes_to_keep: Idx of bounding boxes after applying "Non-max suppression"
            bounding_box_locations: all vec (x, y, w, h) of each chosen bounding box
            class_ids: idx for each predicted class of each bounding box based on COCO dataset's classes
            confidence_values: Probability that the predicted class is correct
    """
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    # Iterate through each layers in YOLOv3 output (totally 3 layers)
    for output in model_outputs:
        # Iterate each bounding boxes in prediction output
        for prediction in output:
            # Getting 80 class's probabilities for current bounding box
            scores = prediction[5:]
            # "class_idx" index of detected object having the highest probability
            class_idx = np.argmax(scores)
            confidence = scores[class_idx]

            # Only detect object having the confident larger than THRESHOLD
            if confidence > THRESHOLD:
                # B.c prediction[2] return between [0-1] --> Need to rescale it to match the position in 320*320 image
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # the center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_idx)
                confidence_values.append(float(confidence))

    # Perform "Non-max suppression" for each prediction bounding boxes
    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values,
                                           THRESHOLD, SUPPRESSION_THRESHOLD)
    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


def show_detected_images(img, bounding_box_ids, all_bounding_boxes, classes, classes_ids,
                         confidence_values, width_ratio, height_ratio):
    """
        Drawing the bounding boxes on the original images
        Args:
            img -- Original image
            bounding_box_ids -- Idx of predicted bounding boxes after applying "Non-max suppression"
            all_bounding_boxes -- all vec (x, y, w, h) of each chosen bounding box
            classes -- list of COCO dataset's classes
            classes_ids -- List of detected class (labels) in the image
            confidence_values -- Probability that the predicted class is correct
            width_ratio = original_width / YOLO_IMAGE_SIZE
            height_ratio = original_height / YOLO_IMAGE_SIZE
            colours -- list of colors for each detected object
    """

    # Iterate each bounding box's idx which is kept after 'non-max suppression'
    for idx in bounding_box_ids.flatten():
        bounding_box = all_bounding_boxes[idx]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        # Transform (x,y,w,h) from training sized image (320*320) to original image size
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        # Color for each detected box
        color_box_current = colors[classes_ids[idx]].tolist()
        # Draw bounding box for each detected object
        cv2.rectangle(img, (x, y), (x+w, y+h), color_box_current, 2)
        # Title for each box
        text_box = classes[int(classes_ids[idx])] + ' ' + str(int(confidence_values[idx] * 100))+'%'
        cv2.putText(img, text_box, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color_box_current, 1)


# Load and get image shape
image = cv2.imread('./data/vung_tau_old.jpg')
original_w, original_h = image.shape[1], image.shape[0]

# Label objects for prediction (totally 80)
with open('./models/coco.names') as f:
    labels = list(line.strip() for line in f)
# Setting colors for each label
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Read the configuration file & initialize the weight of yolov3 model (By OpenCV built-in function)
neural_network = cv2.dnn.readNetFromDarknet('./models/yolov3.cfg', './models/yolov3.weights')

# define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # In case set up backend for GPU
# neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# (NOTE) YOLOv3 model requires BLOB images as the input
# Para explanation:
    # "1/255": for normalization
    # "True" means transform RGB to BGR (B.c cv2 only BGR as input)
# Transform from RGB to BLOB input image
blob = cv2.dnn.blobFromImage(image, 1/255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
neural_network.setInput(blob)  # Set up BLOB image as the input of the model

# Get all the layer's names from model YOLOv3
layers = neural_network.getLayerNames()
# Getting only output layers' names that we need from YOLO v3 algorithm
output_names = \
    [layers[idx[0] - 1] for idx in neural_network.getUnconnectedOutLayers()]

# Apply "Forward propagation" & it results the set of bounding boxes & prediction vector
outputs = neural_network.forward(output_names)
# Output[0].shape: 300 means 300 bounding boxes totally & 85 are prediction vectors
# "85" means 1st 5 parameters (x, y, w, h, conf) + 80 output classes (COCO dataset)

# Extract values from prediction vector
predicted_objects_idx, bbox_locations, class_label_ids, conf_values = find_objects(outputs)

# Draw bounding boxes on original image
show_detected_images(image, predicted_objects_idx, bbox_locations, labels, class_label_ids, conf_values,
                     original_w / YOLO_IMAGE_SIZE, original_h / YOLO_IMAGE_SIZE)

cv2.imshow('YOLO Algorithm', image)
cv2.waitKey()
