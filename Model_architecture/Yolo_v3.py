import tensorflow as tf

# Model Architecture
from Darknet53_Feature_Extraction import darknet53, conv2d_fixed_padding, batch_norm
from Yolo_Convolution_Layer import yolo_convolution_block
from Detection_Layer import detection_layer
from Upsample_Layer import upsample
from Non_Max_Suppression import build_boxes, non_max_suppression

# Model hyper-parameters
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)


class Yolo_v3:
    """Yolo v3 model class"""

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        """Create the model.

        Args:
            n_classes: Number of class labels,
            model_size: the input size of the model
            max_output_size: Max # of boxes to be selected for each class
            iou_threshold: Threshold for IOU
            confidence_threshold: Threshold for the confidence score
            data_format: The input format
        Returns:
            None
        """
        if not data_format:
            # If using CPU for training
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        """Add operations to detect boxes for a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean, whether to use in training or inference mode.

        Returns:
            A list containing class-to-boxes dictionaries
                for each sample in the batch.
    (NOTE) Inference mode: The mode of processing input in a Neural Network wherein the output obtained won't be contributing
            to the gradients and weight updation of the Network.
        """
        with tf.compat.v1.variable_scope('yolo_v3_model'):
            if self.data_format == 'channel_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # NORMALIZATION
            inputs = inputs // 255
            # DARKNET53
            route1, route2, inputs = darknet53(inputs, training=training,
                                               data_format=self.data_format)
            # YOLO CONV BLOCK
            route, inputs = yolo_convolution_block(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            # 1st DETECTION LAYER (1st red part in YOLO architecture)
            detect1 = detection_layer(inputs, n_classes=self.n_classes,
                                      anchors=_ANCHORS[6:9],
                                      img_size=self.model_size,
                                      data_format=self.data_format)

            # CONV2D (fixed pad) - filter=256
            inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            # UPSAMPLE LAYER
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            # Concatenate up-sampling result & route 2 (as follow YOLOv3 architecture)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.keras.layers.concatenate([inputs, route2], axis=axis)
            # YOLO CONV BLOCK
            route, inputs = yolo_convolution_block(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            # 2nd DETECTION LAYER (2nd red part in YOLO architecture)
            detect2 = detection_layer(inputs, n_classes=self.n_classes,
                                      anchors=_ANCHORS[3:6],
                                      img_size=self.model_size,
                                      data_format=self.data_format)

            # CONV2D (fixed pad) - filter=128
            inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            # UPSAMPLE LAYER
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            # Concatenate up-sampling result & route 1 (as follow YOLOv3 architecture)
            inputs = tf.keras.layers.concatenate([inputs, route1], axis=axis)
            # YOLO CONV BLOCK
            route, inputs = yolo_convolution_block(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            # 3rd DETECTION LAYER (3rd red part in YOLO architecture)
            detect3 = detection_layer(inputs, n_classes=self.n_classes,
                                      anchors=_ANCHORS[0:3],
                                      img_size=self.model_size,
                                      data_format=self.data_format)

            # Concatenate all the result of detection layers
            inputs = tf.concat([detect1, detect2, detect3], axis=1)
            # Construct the boxes for detection
            inputs = build_boxes(inputs)
            # NON-MAX SUPPRESSION
            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts