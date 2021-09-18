import tensorflow as tf

_LEAKY_RELU = 0.1
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05

# Description:
# - For features extraction, YOLOv3 use Darknet53 neural network pre-trained model on ImageNet.
# - Same as ResNet model, Darknet53 has shortcut (residual) connections
# - We'll discard the last 3 layers (AvgPool, Connected & Softmax) because we only utilize the features of model


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=1 if data_format == 'channel_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True)(inputs)


def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
    Returns:
        A tensor with the same format as the input.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)(inputs)


def darknet53_residual_block(inputs, filters, training, data_format,
                             strides=1):
    """Creates a residual block for Darknet."""
    shortcut = inputs

    # CONV2D (fixed pad) - kernel_size=1
    inputs = conv2d_fixed_padding(
        inputs, filters, kernel_size=1,
        data_format=data_format, strides=strides
    )
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # CONV2D (fixed pad) - kernel_size=3
    inputs = conv2d_fixed_padding(
        inputs, 2 * filters, kernel_size=3,
        data_format=data_format, strides=strides
    )
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # Residual property
    inputs += shortcut
    return inputs


def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction."""
    # CONV2D (fixed pad) - filter 32
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=32,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # CONV2D (fixed pad) - filter 64
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # DARKNET53 residual block - filter 32
    inputs = darknet53_residual_block(inputs, filters=32, training=training,
                                      data_format=data_format)

    # CONV2D (fixed pad) - filter 128
    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # DARKNET53 residual block *2 - filter 64
    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64,
                                          training=training,
                                          data_format=data_format)

    # CONV2D (fixed pad) - filter 256
    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # DARKNET53 residual block *8 - filter 128
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128,
                                          training=training,
                                          data_format=data_format)
    # ROUTE 1
    route1 = inputs

    # CONV2D (fixed pad) - filter 512
    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # DARKNET53 residual block *8 - filter 256
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256,
                                          training=training,
                                          data_format=data_format)
    # ROUTE 2
    route2 = inputs

    # CONV2D (fixed pad) - filter 1024
    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # DARKNET53 residual block *4 - filter 512
    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512,
                                          training=training,
                                          data_format=data_format)

    return route1, route2, inputs


