import os
import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf


# input and output path 
content_img_path = 'belvidere-castle.jpg' 
style_img_path = 'scream.jpg'
path_output = 'output'


#######################
# algorithm constants #
#######################

# proportion noise to apply to content image
noise_ratio = 0.2
# put emphasis on content loss
alpha = 1
# put emphasis on style loss
beta = 600

# Layers and weights for content and style image
layer_content = [('conv4_2', 1.)]
style_layers = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]


# the mean to subtract from the input to the VGG model
# this is the mean that when the VGG was used to train
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


# the total # of iterations will be outside_iterattions * inside_iterations
outside_iterattions = 10            # number of checkpoints
inside_iterations = 10   # learning iterations per checkpoint


#######################
# algorithm functions #
#######################


def loadRGB(path):
    imgRGB = scipy.misc.imread(path).astype(np.float)
    # convert image to 3D RGB if it is greyscale
    if len(imgRGB.shape)==2:
        w, h = imgRGB.shape
        temp = np.empty((w, h, 3), dtype=np.uint8)
        temp[:, :, 0] = imgRGB
        temp[:, :, 1] = imgRGB
        temp[:, :, 2] = imgRGB
        imgRGB = temp
    return imgRGB


# preprocess funtion for content, style and 'canvas' images
def preprocess(image):
    image = np.reshape(image, ((1,) + image.shape))
    return image - MEAN_VALUES


def save_img(path, img):
    img = img + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image
    img = img[0]
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


# function Return the Conv2D + RELU layer using the weights, biases from the VGG model at 'layer'
def _conv2d_relu(prev_layer, n_layer, layer_name):
    # get weights and bias for current layer:
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    # create a conv2d layer
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
    # add a ReLU function and return
    return tf.nn.relu(conv2d)


# function return average/maximum pooling layer
def pool_layer(pool_style, layer_input):
    if pool_style == 'avg':
        return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif pool_style == 'max':
        return  tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


# content loss
def content_layer_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2 * N * M)) * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def content_loss_func(sess, net):

    layers = layer_content
    total_content_loss = 0.0
    for layer, weight in layers:
        p = sess.run(net[layer])
        x = net[layer]
        total_content_loss += content_layer_loss(p, x)*weight

    total_content_loss /= float(len(layers))
    return total_content_loss


# style loss
def style_layer_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))

    return loss

def gram_matrix(x, area, depth):
    x1 = tf.reshape(x, (area, depth))                   
    G = tf.matmul(tf.transpose(x1), x1)
    return G


def style_loss_func(sess, net):

    layers = style_layers
    total_style_loss = 0.0

    for layer, weight in layers:
        a = sess.run(net[layer])
        x = net[layer]
        total_style_loss += style_layer_loss(a, x) * weight

    total_style_loss /= float(len(layers))

    return total_style_loss


############################################
# preparing output folder and input images #
############################################


# create output directory
if not os.path.exists(path_output):
    os.mkdir(path_output)

# read images and return RGB values
img_content = loadRGB(content_img_path) 
img_style = loadRGB(style_img_path) 

# resize style image to match content
img_style = scipy.misc.imresize(img_style, img_content.shape)

# draw initial canvas with adding noise to content image
noise = np.random.uniform(
        img_content.mean()-img_content.std(), img_content.mean()+img_content.std(),
        (img_content.shape)).astype('float32')
img_initial = noise * noise_ratio + img_content * (1 - noise_ratio)

# preprocess each
img_content = preprocess(img_content)
img_style = preprocess(img_style)
img_initial = preprocess(img_initial)


#####################
# build VGG19 model #
#####################

# with reference to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style


VGG19 = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
VGG19_layers = VGG19['layers'][0]


# Setup network
with tf.Session() as sess:
    a, h, w, d     = img_content.shape
    net = {}
    net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
    net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1')
    net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2')
    net['avgpool1'] = pool_layer('avg',net['conv1_2'])
    net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1')
    net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2')
    net['avgpool2'] = pool_layer('avg',net['conv2_2'])
    net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1')
    net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2')
    net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3')
    net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4')
    net['avgpool3'] = pool_layer('avg',net['conv3_4'])
    net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1')
    net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2')     
    net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3')
    net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4')
    net['avgpool4'] = pool_layer('avg',net['conv4_4'])
    net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1')
    net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2')
    net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3')
    net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4')
    net['avgpool5'] = pool_layer('avg',net['conv5_4'])

        
def main():
    sess.run([net['input'].assign(img_content)])
    content_loss = content_loss_func(sess, net)

    sess.run([net['input'].assign(img_style)])
    style_loss = style_loss_func(sess, net)
    # define loss function and minimise
    total_loss  = alpha * content_loss + beta * style_loss 
    
    # optimizer
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      total_loss, method='L-BFGS-B',
      options={'maxiter': inside_iterations})
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(img_initial))

    for i in range(1,outside_iterattions+1):
        # run optimisation
        optimizer.minimize(sess)

        print('Iteration {}/{}'.format(i*inside_iterations, outside_iterattions*inside_iterations))
        print('total loss:',sess.run(total_loss))

        # write image
        img_output = sess.run(net['input'])
        # generate output images' name
        output_file = path_output+'/'+content_img_path.split('.')[0]+'_'+style_img_path.split('.')[0]+'_'+'%s.jpg' % (i*inside_iterations)
        save_img(output_file, img_output)


if __name__ == '__main__':
    main()
