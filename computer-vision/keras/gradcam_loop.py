#https://github.com/jacobgil/keras-grad-cam

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    #img_path = sys.argv[1]
    img = image.load_img(path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad(y, x):
    V = Lambda(lambda z: K.gradients(
        z[0], z[1]), output_shape=[1])([y, x])
    return V

def grad_cam(input_model, image, category_index, layer_name):

    conv_output = input_model.get_layer(layer_name).output
    y_c = input_model.output[0][0]

    g = grad(y_c, conv_output)[0]
    g = normalize(g)
    g_func = K.function([model.layers[0].input], [conv_output, g])
    
    output, g_val = g_func([image]) #output necessary vs conv_output ??
    output, g_val = output[0, :], g_val[0, :, :, :]
    weights = np.mean(g_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (256, 256))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = 255 * image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
    


from tensorflow.keras.models import load_model
weights_folder = "../resnet-log/"
model = load_model(weights_folder + 'drinking_gray/best.hdf5')
model.summary()

import os
img_folder = "sitting_gray/yes/" #train/yes/
img_list = os.listdir(img_folder) 
for im in img_list:
    preprocessed_input = load_image(img_folder + im)
    predictions = model.predict(preprocessed_input)
    print(predictions)
    if predictions < 0.5:
        predicted_class = int(0)
    else:
        predicted_class = int(1)
    print(predicted_class)

    #conv2_block3_1_relu   conv5_block3_2_relu   conv5_block3_out
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "conv5_block3_out")
    cv2.imwrite(img_folder + "../" + im + "_gradcam.jpg", cam)
    