import tensorflow as tf


def upsample(inputs, out_shape, data_format):
    """
        Upsamples to `out_shape` using nearest neighbor interpolation.
    """
    if data_format == 'channel_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    # "Nearest neighbor interpolation" technique
    inputs = tf.image.resize(inputs, size=(new_height, new_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if data_format == 'channels_first':
        # Transpose back to original shape
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    return inputs

