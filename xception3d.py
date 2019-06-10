"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

# Reference

- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357) (CVPR 2017)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D, UpSampling3D
from tensorflow.keras.layers import DepthwiseConv2D
# from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding3D
# from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import AveragePooling3D, MaxPooling3D, add, Dense
# from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.data_utils import get_file

# import os
# import warnings

# from . import get_submodules_from_kwargs
# from . import imagenet_utils
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import _obtain_input_shape


# TF_WEIGHTS_PATH = (
#     'https://github.com/fchollet/deep-learning-models/'
#     'releases/download/v0.4/'
#     'xception_weights_tf_dim_ordering_tf_kernels.h5')
# TF_WEIGHTS_PATH_NO_TOP = (
#     'https://github.com/fchollet/deep-learning-models/'
#     'releases/download/v0.4/'
#     'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def Xception(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    # SepConv3D_counter = 0

    def SepConv3D(filter, x):
      # global SepConv3D_counter
      x = Conv3D(filter, (3, 3, 3),
                           padding='same',
                           use_bias=False)(x)
      # SepConv3D_counter += 1
      return x

    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=299,
    #                                   min_size=71,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    channel_axis = -1

    x = Conv3D(32, (3, 3, 3),
                      strides=(2, 2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv3D(64, (3, 3, 3), use_bias=False, name='block1_conv2')(x)

    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv3D(128, (1, 1, 1),
                             strides=(2, 2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = SepConv3D(128, x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)

    x = SepConv3D(128, x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling3D((3, 3, 3),
                            strides=(2, 2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv3D(128, (1, 1, 1), strides=(2, 2, 2),# 256
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)

    x = SepConv3D(128, x)# 256
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)

    x = SepConv3D(128, x)# 256
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2),
                            padding='same',
                            name='block3_pool')(x)

    x = add([x, residual])

    residual = Conv3D(128, (1, 1, 1),# 728
                             strides=(2, 2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SepConv3D(128, x)# 728
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SepConv3D(128, x)# 728
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SepConv3D(128, x)# 728
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SepConv3D(128, x)# 728
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SepConv3D(128, x)# 728
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv3D(128, (1, 1, 1), strides=(2, 2, 2),# 1024
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SepConv3D(128, x)# 728
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SepConv3D(128, x)# 1024
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling3D((3, 3, 3),
                            strides=(2, 2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = add([x, residual])

    x = SepConv3D(128, x)# 1536
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SepConv3D(128, x)# 2048
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling3D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='xception')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)