# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:37:22 2022

@author: filon
"""

import time
import imageio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

tf.compat.v1.disable_eager_execution()
%matplotlib inline

# path to image
target_path = './Images/image.jpg'

# style image path
style_path = "./Images/style.jpg"

result_prefix = style_path.split('Images/')[1][:-4] + '_onto_' + target_path.split('Images/')[1][:-4]

# Image dimensions
w, h = load_img(target_path).size
img_h = 400
img_w = int(w + img_h/h)

def preprocess_img(img_path):
    img = load_img(img_path, target_size = (img_h, img_w))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_img(x):
    # Remove zero-center bu mean pixel and adding standardization values to B, G, R channels
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # BGR -> RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8') #limits the values of x between 0 and 255
    return x

def content_loss(target, final):
    return K.sum(K.square(target-final))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, final):
    S = gram_matrix(style)
    F = gram_matrix(final)
    channels = 3
    size = img_h + img_w
    return K.sum(K.square(S - F)) / (4.*(channels**2)*(size**2))

def total_vairation_loss(x):
    a = K.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, 1:, :img_w - 1, :])
    b = K.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, :img_h - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# load reference images and style image
target_img = K.constant(preprocess_img(target_path))
style_img = K.constant(preprocess_img(style_path))

# placeholder will containt final generated image
final_img = K.placeholder((1, img_h, img_w, 3))

# combine 3 images into a single batch
input_tensor = K.concatenate([target_img, style_img, final_img], axis = 0)

# build the vgg16 network with batch of images as input
# model is loaded with pretrained ImageNet weights
model = vgg16.VGG16(input_tensor = input_tensor, weights = 'imagenet', include_top = False)

print('Model loaded')

# create a dictionary containing layer_name:layer_output
output_dict = dict([(layer.name, layer.output) for layer in model.layers])

# name of layer used for content loss
content_layer = 'block5_conv2'

# name f layers used for style loss
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# weights in the weighted average of the loss components
total_vary_weight = 1e-4 # random number
style_weight = 1 # random number
content_weight = 0.025 # random number

# define the loss by adding all components to a loss variable
loss = K.variable(0.)
layer_features = output_dict[content_layer]
target_img_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight*content_loss(target_img_features, combination_features)

for layer_name in style_layers:
    layer_features = output_dict[layer_name]
    style_img_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    sl = style_loss(style_img_features, combination_features)
    loss += sl*(style_weight/len(style_layers))
    
loss += total_vary_weight*total_vairation_loss(final_img)

# get the gradient of the loss with the final imgage shows how loss is changing with final image
grads = K.gradients(loss, final_img)[0]

# Function to fetch the values of the current loss and current gradient
fetch_loss_grad = K.function([final_img], [loss, grads])

class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_h, img_w, 3))
        outs = fetch_loss_grad([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# After 10 iterations little change occurs
iter = 50

# run scipy-based optimization over pixels of the generated image so as to minimize the neural style loss
# this is our initial state: the target image
# note that scipy.optimize.fmin_1_bfgs_b can only process flat vectors
x = preprocess_img(target_path)
x = x.flatten()

# fmin_1_bfgs_b(func, x) minmizes a function func using the L-BFGS-B allgorithm where
# x is the intial gyess
# fprime is the gradient of the function
# maxfun is max number off function evals

# returns x which is estimated position of the minimum
# minval -> value of func at the minimum

for i in range(iter):
    print('Start of iteration', i)
    start_time = time.time()
    estimated_min, func_val_at_min, info = fmin_l_bfgs_b(evaluator.loss, x, fprime = evaluator.grads, maxfun = 20)
    print('Current loss value:', func_val_at_min)
    
    # save generated images
    img = estimated_min.copy().reshape((img_h, img_w, 3))
    img = deprocess_img(img)
    fname = "./Output/name" + result_prefix + '_at_iterations_%d.jpg' % i
    imageio.imwrite(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    print('\n')
    
plt.figure(figsize = (15,8))

# content image
plt.subplot(131)
plt.title('Content Image')
plt.imshow(load_img(target_path, target_size = (img_h, img_w)))

# style image
plt.subplot(132)
plt.title('Style Image')
plt.imshow(load_img(style_path, target_size = (img_h, img_w)))

# content image
plt.subplot(133)
plt.title('Generated Image')
plt.imshow(img)
