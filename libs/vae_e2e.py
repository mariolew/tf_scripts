"""Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Parag K. Mital, January 2016
"""
import tensorflow as tf
import numpy as np
import os
import pickle
from libs.dataset_utils import create_input_pipeline
#from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.tfpipeline import input_pipeline
from libs.tfpipeline import input_pipeline_reg
from libs.tfpipeline import input_pipeline_reg_test
from libs.tfpipeline import input_pipeline_local

def VAE(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False):
    """(Variational) (Convolutional) (Denoising) Autoencoder.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    n_hidden : int, optional
        Only applied when variational=True.  This refers to the first fully
        connected layer prior to the variational embedding, directly after
        the encoding.  After the variational embedding, another fully connected
        layer is created with the same size prior to decoding.  Set to 0 to
        not use an additional hidden layer.
    n_code : int, optional
        Only applied when variational=True.  This refers to the number of
        latent Gaussians to sample for creating the inner most encoding.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.nn.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.
    denoising : bool, optional
        Whether or not to apply denoising.  If using denoising, you must feed a
        value for 'corrupt_prob', as returned in the dictionary.  1.0 means no
        corruption is used.  0.0 means every feature is corrupted.  Sensible
        values are between 0.5-0.8.
    convolutional : bool, optional
        Whether or not to use a convolutional network or else a fully connected
        network will be created.  This effects the n_filters parameter's
        meaning.
    variational : bool, optional
        Whether or not to create a variational embedding layer.  This will
        create a fully connected layer after the encoding, if `n_hidden` is
        greater than 0, then will create a multivariate gaussian sampling
        layer, then another fully connected layer.  The size of the fully
        connected layers are determined by `n_hidden`, and the size of the
        sampling layer is determined by `n_code`.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'keep_prob': Amount to keep when using Dropout
            'corrupt_prob': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    label = tf.placeholder(tf.float32, [None, 64, 64, 68], 'y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])

    if denoising:
        current_input = utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x) if convolutional else x
    current_input = x_tensor

    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input.get_shape().as_list()
            flattened = utils.flatten(current_input)

            if n_hidden:
                h = utils.linear(flattened, n_hidden, name='W_fc')[0]
                #h = activation(batch_norm(h, phase_train, 'fc/bn'))
                h = activation(h)
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = flattened

            z_mu = utils.linear(h, n_code, name='mu')[0]
            z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x)[0], n_code]))

            # Sample from posterior
            z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))

            if n_hidden:
                h = utils.linear(z, n_hidden, name='fc_t')[0]
                h = activation(h)
                #h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = z

            size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
            h = utils.linear(h, size, name='fc_t2')[0]
            #current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            current_input = activation(h)
            if dropout:
                current_input = tf.nn.dropout(current_input, keep_prob)

            if convolutional:
                current_input = tf.reshape(
                    current_input, tf.stack([
                        tf.shape(current_input)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h
    current_input, W = utils.conv2d(current_input, 68, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='final_conv')

    y = current_input
    x_flat = utils.flatten(label)
    
    y = tf.nn.sigmoid(y)
    y_flat = utils.flatten(y)
    # l2 loss
    loss_x = -tf.reduce_sum( (  (x_flat*tf.log(y_flat + 1e-9)) + ((1-x_flat) * tf.log(1 - y_flat + 1e-9))), 1  , name='xentropy' )
    #loss_x = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_flat, x_flat), 1)

    if variational:
        # variational lower bound, kl-divergence
        loss_z = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma -
            tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)

        # add l2 loss
        cost = tf.reduce_mean(loss_x + loss_z)
    else:
        # just optimize l2 loss
        cost = tf.reduce_mean(loss_x)

    return {'cost': cost, 'Ws': Ws, 'label': label,
            'x': x, 'z': z, 'y': y,
            'keep_prob': keep_prob,
            'corrupt_prob': corrupt_prob,
            'train': phase_train}
def VAE_ALIGN(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False):
    """(Variational) (Convolutional) (Denoising) Autoencoder.

    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    label = tf.placeholder(tf.float32, [None, 136], 'y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    phase_keep = tf.placeholder(tf.bool, name='phase_keep')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])

    if denoising:
        current_input = utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x) if convolutional else x
    current_input = x_tensor

    Ws = []
    shapes = []
    hs = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            hs.append(h)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input.get_shape().as_list()
            flattened = utils.flatten(current_input)

            if n_hidden:
                h = utils.linear(flattened, n_hidden, name='W_fc')[0]
                #h = activation(batch_norm(h, phase_train, 'fc/bn'))
                h = activation(h)
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = flattened

            z_mu = utils.linear(h, n_code, name='mu')[0]
            z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x)[0], n_code]))

            # Sample from posterior
            if dropout:
                z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))
            else:
                z = z_mu

            if n_hidden:
                h = utils.linear(z, n_hidden, name='fc_t')[0]
                h = activation(h)
                #h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = z

            size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
            h = utils.linear(h, size, name='fc_t2')[0]
            #current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            current_input = activation(h)
            if dropout:
                current_input = tf.nn.dropout(current_input, keep_prob)

            if convolutional:
                current_input = tf.reshape(
                    current_input, tf.stack([
                        tf.shape(current_input)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h
    current_input, W = utils.conv2d(current_input, 68, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='final_conv')

    y = tf.nn.sigmoid(current_input)
    h = tf.concat_v2([hs[0], y], 3)

    with tf.variable_scope('align/'):
        #h = tf.concat(3, [batch_norm(hs[0], phase_train, affine=False), y])
        h, W = utils.conv2d(h, 48, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv1')
        h = activation(batch_norm(h, phase_train, 'bn1'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
        loss1h, wloss1 = utils.linear(h, 136, name='loss1')

        h, W = utils.conv2d(h, 64, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv2')
        h = activation(batch_norm(h, phase_train, 'bn2'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
        loss2h, wloss2 = utils.linear(h, 136, name='loss2')

        h, W = utils.conv2d(h, 96, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv3')
        h3 = activation(batch_norm(h, phase_train, 'bn3'))
        #h3 = activation(h)
        if dropout:
            h = tf.nn.dropout(h3, keep_prob)
        loss3h, wloss3 = utils.linear(h, 136, name='loss3')

        h, W = utils.conv2d(h3, 128, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv4')
        h = activation(batch_norm(h, phase_train, 'bn4'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
        #h = activation(h)
        h3_flat = utils.flatten(h3)
        h_flat = utils.flatten(h)
        concat = tf.concat_v2([h3_flat, h_flat], 1)
        ip1, W1 = utils.linear(concat, 256, name='ip1')
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob)
        ip2, W2 = utils.linear(ip1, 192, name='ip2')
        ip2 = activation(ip2)
        if dropout:
            ip2 = tf.nn.dropout(ip2, keep_prob)
        ip3, W3 = utils.linear(ip2, 136, name='ip3')

    p_flat = utils.flatten(ip3)
    y_flat = utils.flatten(label)
    regularizers = 5e-4 *(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    loss_x = tf.reduce_sum(tf.squared_difference(p_flat, y_flat), 1)
    loss1 = tf.reduce_sum(tf.squared_difference(loss1h, y_flat), 1)
    loss2 = tf.reduce_sum(tf.squared_difference(loss2h, y_flat), 1)
    loss3 = tf.reduce_sum(tf.squared_difference(loss3h, y_flat), 1)
    lab = tf.reshape(y_flat, (-1, 68, 2))
    norm = tf.reduce_sum(((lab[:, 36, :] - lab[:, 45, :])**2), 1)
    cost = tf.reduce_mean((loss_x + loss1 + loss2 + loss3)/(norm)) + regularizers
    prediction = tf.reshape(p_flat, (-1, 68, 2))

    return {'cost': cost, 'Ws': Ws, 'label': label,
            'x': x, 'z': z, 'y': prediction,
            'keep_prob': keep_prob,
            'corrupt_prob': corrupt_prob,
            'train': phase_train,
            'keep': phase_keep}

def VAE_ALIGN1(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False):
    """(Variational) (Convolutional) (Denoising) Autoencoder.

    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    #landMaps = tf.placeholder(tf.float32, [None, 64, 64, 68], 'lmaps')
    label = tf.placeholder(tf.float32, [None, 136], 'y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    phase_keep = tf.placeholder(tf.bool, name='phase_keep')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])
    keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
    keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')
    if denoising:
        current_input = utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x) if convolutional else x
    current_input = x_tensor

    Ws = []
    shapes = []
    hs = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            hs.append(h)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input.get_shape().as_list()
            flattened = utils.flatten(current_input)

            if n_hidden:
                h = utils.linear(flattened, n_hidden, name='W_fc')[0]
                #h = activation(batch_norm(h, phase_train, 'fc/bn'))
                h = activation(h)
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = flattened

            z_mu = utils.linear(h, n_code, name='mu')[0]
            z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x)[0], n_code]))

            # Sample from posterior
            if dropout:
                z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))
            else:
                z = z_mu

            if n_hidden:
                h = utils.linear(z, n_hidden, name='fc_t')[0]
                h = activation(h)
                #h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = z

            size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
            h = utils.linear(h, size, name='fc_t2')[0]
            #current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            current_input = activation(h)
            if dropout:
                current_input = tf.nn.dropout(current_input, keep_prob)

            if convolutional:
                current_input = tf.reshape(
                    current_input, tf.stack([
                        tf.shape(current_input)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            #h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            h = activation(h)
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h
    current_input, W = utils.conv2d(current_input, 68, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='final_conv')

    y = tf.nn.sigmoid(current_input)
    h = tf.concat_v2([hs[0], y], 3)
    #x_flat = utils.flatten(landMaps)
    #y_flat = utils.flatten(y)
    #loss_x = -tf.reduce_sum( (  (x_flat*tf.log(y_flat + 1e-9)) + ((1-x_flat) * tf.log(1 - y_flat + 1e-9))), 1  , name='xentropy' )
    #loss_x = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_flat, x_flat), 1)

    #loss_z = -0.5 * tf.reduce_sum(
    #        1.0 + 2.0 * z_log_sigma -
    #        tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)
    #cost_vae = tf.reduce_mean(loss_x + loss_z)
    with tf.variable_scope('align/'):
        #h = tf.concat(3, [batch_norm(hs[0], phase_train, affine=False), y])
        h, W = utils.conv2d(h, 48, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv1')
        h = activation(batch_norm(h, phase_train, 'bn1'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob1)
        h, W = utils.conv2d(h, 48, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv11')
        h = activation(batch_norm(h, phase_train, 'bn11'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob1)
        # loss 1
        #loss1h, wloss1 = utils.linear(h, 136, name='loss1')

        h, W = utils.conv2d(h, 64, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv2')
        h = activation(batch_norm(h, phase_train, 'bn2'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob1)
        h, W = utils.conv2d(h, 64, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv22')
        h = activation(batch_norm(h, phase_train, 'bn22'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob1)
        # loss 2
        #loss2h, wloss2 = utils.linear(h, 136, name='loss2')

        h, W = utils.conv2d(h, 96, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv3')
        h = activation(batch_norm(h, phase_train, 'bn3'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob)

        h, W = utils.conv2d(h, 96, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv33')
        h3 = activation(batch_norm(h, phase_train, 'bn33'))
        #h3 = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
        # loss 3
        #loss3h, wloss3 = utils.linear(h, 136, name='loss3')

        h, W = utils.conv2d(h3, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv4')
        h = activation(batch_norm(h, phase_train, 'bn4'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob2)
        h, W = utils.conv2d(h3, 128, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name='conv44')
        h = activation(batch_norm(h, phase_train, 'bn44'))
        #h = activation(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob2)
        #h = activation(h)
        h3_flat = utils.flatten(h3)
        h_flat = utils.flatten(h)
        concat = tf.concat_v2([h3_flat, h_flat], 1)
        ip1, W1 = utils.linear(concat, 256, name='ip1')
        ip1 = activation(ip1)
        if dropout:
            ip1 = tf.nn.dropout(ip1, keep_prob1)
        ip2, W2 = utils.linear(ip1, 192, name='ip2')
        ip2 = activation(ip2)
        if dropout:
            ip2 = tf.nn.dropout(ip2, keep_prob1)
        ip3, W3 = utils.linear(ip2, 136, name='ip3')

    p_flat = utils.flatten(ip3)
    y_flat = utils.flatten(label)
    regularizers = 5e-4 *(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    loss_x = tf.reduce_sum(tf.squared_difference(p_flat, y_flat), 1)
    #loss1 = tf.reduce_sum(tf.squared_difference(loss1h, y_flat), 1)
    #loss2 = tf.reduce_sum(tf.squared_difference(loss2h, y_flat), 1)
    #loss3 = tf.reduce_sum(tf.squared_difference(loss3h, y_flat), 1)
    lab = tf.reshape(y_flat, (-1, 68, 2))
    norm = tf.reduce_sum(((lab[:, 36, :] - lab[:, 45, :])**2), 1)
    cost = tf.reduce_mean((loss_x)/(norm)) + regularizers# + cost_vae
    prediction = tf.reshape(p_flat, (-1, 68, 2))

    return {'cost': cost, 'Ws': Ws, 'label': label,
            'x': x, 'z': z, 'y': prediction,
            'keep_prob': keep_prob,
            'keep_prob1': keep_prob1,
            'keep_prob2': keep_prob2,
            'corrupt_prob': corrupt_prob,
            'train': phase_train,
            'keep': phase_keep}

def eval_vae_align(files,
              input_shape,
              batch_size=64,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=1,
              save_step=5000,
              ckpt_name="vae.ckpt"):
    
    
    batch = input_pipeline_reg_test(['300w-gt-test.txt'], batch_size=batch_size, shape=[128, 128, 1], is_training=False)
    ae = VAE_ALIGN1(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    
    # opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("align/")]
    # ori_vars = [v for v in tf.all_variables() if not v.name.startswith("align/")]
    # batch_idx = tf.Variable(0, dtype=tf.int32)
    # learning_rate = tf.train.exponential_decay(0.0, batch_idx * batch_size, 150000, 0.95, staircase=True)
    # optimizer = tf.train.AdamOptimizer(
    #     learning_rate=learning_rate).minimize(ae['cost'], var_list=opt_vars, global_step=batch_idx)

    avg_pred = ae['y']
    gt_truth = ae['label']
    gt_truth = tf.reshape(gt_truth, (-1, 68, 2))
    # Calculate predictions.
    norm_error = utils.normalized_rmse(avg_pred, gt_truth)
    # We create a session to use the graph
    sess = tf.Session()
    saver = tf.train.Saver()
    # saver_m = tf.train.Saver(tf.all_variables())
    # sess.run(tf.initialize_all_variables())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()


    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name+'.data-00000-of-00001'):
        saver.restore(sess, ckpt_name)
        print('load ' + ckpt_name + ' successfully')


    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 10 
    #test_xs, test_label = sess.run(batch)
    # test_xs = test_xs
    #print(test_xs.max())
    # utils.montage_landmarks(test_label[:8], 'map_train/test_xs.png')
    all_err = []
    fucking_err = []
    predictions = []
    gts = []
    
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, label_xs = sess.run(batch)
            #batch_xs = batch_xs
            train_cost, pred, norm_err = sess.run([ae['cost'], ae['y'], norm_error], feed_dict={
                ae['x']: batch_xs, ae['label']: label_xs, ae['train']: False,
                ae['keep_prob']: keep_prob, ae['keep_prob1']: 0.9, ae['keep_prob2']: 0.7, ae['keep']: False})[:3]

            # Directly from map
             
            #flipped = np.asarray([utils.flip_img(batch_xs[0])])
            #import pdb; pdb.set_trace()
            #pred_flip = sess.run(ae['y'], feed_dict={ae['x']:flipped, ae['train']: False, ae['keep_prob']: keep_prob, ae['keep']: False})
            #import pdb; pdb.set_trace()
            #pred_flip = np.asarray([utils.flip_landmarks(pred_flip[0])])
            predictions.append(pred)
            gts.append(label_xs[0].reshape([68, 2]))
            print(batch_i, train_cost)
            cost += train_cost
            if batch_i % n_files == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:

                label_xs = label_xs.reshape([-1, 68, 2])
                err = utils.evaluateBatchError(label_xs, pred, batch_size)
                all_err.append(err)
                fucking_err.append(norm_err)
                print('Mean error:' + np.array_str(err))
                #print(norm_err)
                t_i += 1


    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        all_err = np.asarray(all_err)
        fucking_err = np.vstack(fucking_err).ravel()
        auc_at_08 = (fucking_err < .08).mean()
        auc_at_05 = (fucking_err < .05).mean()
        #output = open('fullset-det-gt-0109.pkl', 'wb')
        #pickle.dump(predictions, output)
        #output.close()

        print('Overall mean error: ' + np.array_str(all_err.mean(axis=0)))
        print('LFPW: %f, HELEN: %f, IBUG: %f' %(all_err[:224].mean(), all_err[224:554].mean(), all_err[554:].mean()))
        print('Fucking error: %f, auc@05: %.4f, auc@08: %.4f.' %(fucking_err.mean(), auc_at_05, auc_at_08))
        # saver_m.save(sess, "./" + "align",
        #     global_step=batch_i,
        #     write_meta_graph=False)
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

def train_vae_align(files,
              input_shape,
              batch_size=64,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=20000,
              ckpt_name="vae.ckpt"):
    
    
    batch = input_pipeline_reg(['300w-gt-aug.txt'], batch_size=64, shape=[128, 128, 1], is_training=True)
    ae = VAE_ALIGN1(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    
    opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("align/")]
    ori_vars = [v for v in tf.global_variables() if not v.name.startswith("align/")]
    #old_names = ["align/bn1/align/bn1/moments/moments_1/mean/ExponentialMovingAverage", "align/bn11/align/bn11/moments/moments_1/mean/ExponentialMovingAverage",
    #"align/bn2/align/bn2/moments/moments_1/mean/ExponentialMovingAverage", "align/bn22/align/bn22/moments/moments_1/mean/ExponentialMovingAverage",
    #"align/bn3/align/bn3/moments/moments_1/mean/ExponentialMovingAverage", "align/bn33/align/bn33/moments/moments_1/mean/ExponentialMovingAverage",
    #"align/bn4/align/bn4/moments/moments_1/mean/ExponentialMovingAverage",
    #"align/bn1/align/bn1/moments/moments_1/variance/ExponentialMovingAverage", "align/bn11/align/bn11/moments/moments_1/variance/ExponentialMovingAverage",
    #"align/bn2/align/bn2/moments/moments_1/variance/ExponentialMovingAverage", "align/bn22/align/bn22/moments/moments_1/variance/ExponentialMovingAverage",
    #"align/bn3/align/bn3/moments/moments_1/variance/ExponentialMovingAverage", "align/bn33/align/bn33/moments/moments_1/variance/ExponentialMovingAverage",
    #"align/bn4/align/bn4/moments/moments_1/variance/ExponentialMovingAverage"]
    #new_names = ["align//bn1/align/bn1/moments/moments_1/mean/ExponentialMovingAverage", "align//bn11/align/bn11/moments/moments_1/mean/ExponentialMovingAverage",
    #"align//bn2/align/bn2/moments/moments_1/mean/ExponentialMovingAverage", "align//bn22/align/bn22/moments/moments_1/mean/ExponentialMovingAverage",
    #"align//bn3/align/bn3/moments/moments_1/mean/ExponentialMovingAverage", "align//bn33/align/bn33/moments/moments_1/mean/ExponentialMovingAverage",
    #"align//bn4/align/bn4/moments/moments_1/mean/ExponentialMovingAverage",
    #"align//bn1/align/bn1/moments/moments_1/variance/ExponentialMovingAverage", "align//bn11/align/bn11/moments/moments_1/variance/ExponentialMovingAverage",
    #"align//bn2/align/bn2/moments/moments_1/variance/ExponentialMovingAverage", "align//bn22/align/bn22/moments/moments_1/variance/ExponentialMovingAverage",
    #"align//bn3/align/bn3/moments/moments_1/variance/ExponentialMovingAverage", "align//bn33/align/bn33/moments/moments_1/variance/ExponentialMovingAverage",
    #"align//bn4/align/bn4/moments/moments_1/variance/ExponentialMovingAverage"]
    names_to_vars = {v.op.name: v for v in tf.global_variables()}
    #import pdb; pdb.set_trace()
    #for (i,new_name) in enumerate(new_names):
    #    bias_var = names_to_vars[new_name]
    #    names_to_vars[old_names[i]] = bias_var
    #    del names_to_vars[new_name]
    #import pdb; pdb.set_trace()
    batch_idx = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(0.0006, batch_idx * batch_size, 192000, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'], global_step=batch_idx)

    # We create a session to use the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    saver = tf.train.Saver(ori_vars)
    saver_m = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print('load ' + ckpt_name + ' successfully')
    print(learning_rate)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 100000
    # utils.montage_landmarks(test_label[:8], 'map_train/test_xs.png')
    # all_err = []
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, label_xs = sess.run(batch)
            #batch_xs = batch_xs
            train_cost, pred = sess.run([ae['cost'], ae['y'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['label']: label_xs, ae['train']: True,
                ae['keep_prob']: keep_prob, ae['keep_prob1']: 0.9, ae['keep_prob2']: 0.7, ae['keep']: False})[:2]
            print(batch_i, train_cost)
            cost += train_cost
            if batch_i % n_files == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:
                lr = sess.run(learning_rate)
                print('learning rate: %9f' %lr)
                label_xs = label_xs.reshape([-1, 68, 2])
                err = utils.evaluateBatchError(label_xs, pred, batch_size)
                # all_err.append(err)
                print('Mean error:' + np.array_str(err))
                
                t_i += 1

            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver_m.save(sess, "models_e2e/" + "align-300w-gtbbx",
                           global_step=batch_i,
                           write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        # all_err = np.asarray(all_err)

        # print('mean error:' + np.array_str(all_err.mean(axis=0)))
        saver_m.save(sess, "models_e2e/" + "align-300w-gtbbx",
             global_step=t_i,
             write_meta_graph=False)
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

def train_vae_align1(files,
              input_shape,
              batch_size=64,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=20000,
              ckpt_name="vae.ckpt"):
    
    
    batch = input_pipeline_reg(['300w-gt-aug.txt'], batch_size=64, shape=[128, 128, 1], is_training=True)
    ae = VAE_ALIGN(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    
    opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("align/")]
    ori_vars = [v for v in tf.global_variables() if not v.name.startswith("align/")]
    batch_idx = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(0.0003, batch_idx * batch_size, 192000, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'], var_list=opt_vars, global_step=batch_idx)

    # We create a session to use the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    saver = tf.train.Saver(ori_vars)
    saver_m = tf.train.Saver(tf.global_variables())
    sess.run(tf.tf.global_variables_initializer())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print('load ' + ckpt_name + ' successfully')
    print(learning_rate)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 100000
    test_xs, test_label = sess.run(batch)
    test_xs = test_xs
    print(test_xs.max())
    # utils.montage_landmarks(test_label[:8], 'map_train/test_xs.png')
    # all_err = []
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, label_xs = sess.run(batch)
            #batch_xs = batch_xs
            train_cost, pred = sess.run([ae['cost'], ae['y'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['label']: label_xs, ae['train']: True,
                ae['keep_prob']: keep_prob, ae['keep']: False})[:2]
            print(batch_i, train_cost)
            cost += train_cost
            if batch_i % n_files == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:
                lr = sess.run(learning_rate)
                print('learning rate: %9f' %lr)
                label_xs = label_xs.reshape([-1, 68, 2])
                err = utils.evaluateBatchError(label_xs, pred, batch_size)
                # all_err.append(err)
                print('Mean error:' + np.array_str(err))
                
                t_i += 1

            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver_m.save(sess, "/mnt/dataset2/tea/tfmodels/" + "align-300w-gtbbx-new",
                           global_step=batch_i,
                           write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        # all_err = np.asarray(all_err)

        # print('mean error:' + np.array_str(all_err.mean(axis=0)))
        saver_m.save(sess, "/mnt/dataset2/tea/tfmodels/" + "align-300w-gtbbx-new",
             global_step=t_i,
             write_meta_graph=False)
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

def eval_vae(files,
              input_shape,
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              ckpt_name="vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
        List of paths to images.
    input_shape : list
        Must define what the input image's shape is.
    learning_rate : float, optional
        Learning rate.
    batch_size : int, optional
        Batch size.
    n_epochs : int, optional
        Number of epochs.
    n_examples : int, optional
        Number of example to use while demonstrating the current training
        iteration's reconstruction.  Creates a square montage, so make
        sure int(sqrt(n_examples))**2 = n_examples, e.g. 16, 25, 36, ... 100.
    crop_shape : list, optional
        Size to centrally crop the image to.
    crop_factor : float, optional
        Resize factor to apply before cropping.
    n_filters : list, optional
        Same as VAE's n_filters.
    n_hidden : int, optional
        Same as VAE's n_hidden.
    n_code : int, optional
        Same as VAE's n_code.
    convolutional : bool, optional
        Use convolution or not.
    variational : bool, optional
        Use variational layer or not.
    filter_sizes : list, optional
        Same as VAE's filter_sizes.
    dropout : bool, optional
        Use dropout or not
    keep_prob : float, optional
        Percent of keep for dropout.
    activation : function, optional
        Which activation function to use.
    img_step : int, optional
        How often to save training images showing the manifold and
        reconstruction.
    save_step : int, optional
        How often to save checkpoints.
    ckpt_name : str, optional
        Checkpoints will be named as this, e.g. 'model.ckpt'
    """
    #batch = create_input_pipeline(
    #    files=files,
    #    batch_size=batch_size,
    #    n_epochs=n_epochs,
    #    crop_shape=crop_shape,
    #    crop_factor=crop_factor,
    #    shape=input_shape)
    
    batch = input_pipeline(['tftest_vae.txt'], batch_size=batch_size, shape=[64, 64, 1], is_training=True)
    ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)



    # We create a session to use the graph
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print('load' + ckpt_name + 'successfully')

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 1000
    test_xs, test_label = sess.run(batch)
    test_xs = test_xs
    print(test_xs.max())
    utils.montage_landmarks(test_label[:8], 'map_test/test_xs.png')
    try:
        #batch_xs, label_xs = sess.run(batch)


        # Plot example reconstructions
        recon = sess.run(
            ae['y'], feed_dict={ae['x']: test_xs,
                                ae['train']: False,
                                ae['keep_prob']: 1.0})
        print('reconstruction (min, max, mean):',
            recon.min(), recon.max(), recon.mean())
        utils.montage_landmarks(recon[:8],
                      'map_test/reconstruction.png')


    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

def train_vae(files,
              input_shape,
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=200,
              save_step=20000,
              ckpt_name="vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
        List of paths to images.
    input_shape : list
        Must define what the input image's shape is.
    learning_rate : float, optional
        Learning rate.
    batch_size : int, optional
        Batch size.
    n_epochs : int, optional
        Number of epochs.
    n_examples : int, optional
        Number of example to use while demonstrating the current training
        iteration's reconstruction.  Creates a square montage, so make
        sure int(sqrt(n_examples))**2 = n_examples, e.g. 16, 25, 36, ... 100.
    crop_shape : list, optional
        Size to centrally crop the image to.
    crop_factor : float, optional
        Resize factor to apply before cropping.
    n_filters : list, optional
        Same as VAE's n_filters.
    n_hidden : int, optional
        Same as VAE's n_hidden.
    n_code : int, optional
        Same as VAE's n_code.
    convolutional : bool, optional
        Use convolution or not.
    variational : bool, optional
        Use variational layer or not.
    filter_sizes : list, optional
        Same as VAE's filter_sizes.
    dropout : bool, optional
        Use dropout or not
    keep_prob : float, optional
        Percent of keep for dropout.
    activation : function, optional
        Which activation function to use.
    img_step : int, optional
        How often to save training images showing the manifold and
        reconstruction.
    save_step : int, optional
        How often to save checkpoints.
    ckpt_name : str, optional
        Checkpoints will be named as this, e.g. 'model.ckpt'
    """
    #batch = create_input_pipeline(
    #    files=files,
    #    batch_size=batch_size,
    #    n_epochs=n_epochs,
    #    crop_shape=crop_shape,
    #    crop_factor=crop_factor,
    #    shape=input_shape)
    
    batch = input_pipeline(['300w-gt-aug.txt'], batch_size=batch_size, shape=[128, 128, 1], is_training=True)
    ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print('Successfully load '+ckpt_name)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 100000 
    test_xs, test_label = sess.run(batch)
    test_xs = test_xs
    print(test_xs.max())
    utils.montage_landmarks(test_label[0:8], 'map_train/test_xs.png')
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, label_xs = sess.run(batch)

            train_cost = sess.run([ae['cost'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['label']: label_xs, ae['train']: True,
                ae['keep_prob']: keep_prob})[0]
            print(batch_i, train_cost)
            cost += train_cost
            if batch_i % n_files == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:
                # Plot example reconstructions from latent layer
            #    recon = sess.run(
            #        ae['y'], feed_dict={
            #            ae['z']: zs,
            #            ae['train']: False,
            #            ae['keep_prob']: 1.0})
            #    utils.montage_landmarks(recon[:8],
            #                  'map_train/manifold_%08d.png' % t_i)

                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0})
                print('reconstruction (min, max, mean):',
                    recon.min(), recon.max(), recon.mean())
                utils.montage_landmarks(recon[:8],
                              'map_train/reconstruction_%08d.png' % t_i)
                t_i += 1

            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver.save(sess, "models/" + 'vae_gt%d'%batch_i,
                           global_step=batch_i,
                           write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        saver.save(sess, "models/" + 'vae_shit_gt',
                           global_step=batch_i,
                           write_meta_graph=False)
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()


# %%
def test_mnist():
    """Train an autoencoder on MNIST.

    This function will train an autoencoder on MNIST and also
    save many image files during the training process, demonstrating
    the latent space of the inner most dimension of the encoder,
    as well as reconstructions of the decoder.
    """

    # load MNIST
    n_code = 2
    mnist = MNIST(split=[0.8, 0.1, 0.1])
    ae = VAE(input_shape=[None, 784], n_filters=[512, 256],
             n_hidden=64, n_code=n_code, activation=tf.nn.sigmoid,
             convolutional=False, variational=True)

    n_examples = 100
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    learning_rate = 0.02
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    t_i = 0
    batch_i = 0
    batch_size = 200
    n_epochs = 10
    test_xs = mnist.test.images[:n_examples]
    utils.montage(test_xs.reshape((-1, 28, 28)), 'test_xs.png')
    for epoch_i in range(n_epochs):
        train_i = 0
        train_cost = 0
        for batch_xs, _ in mnist.train.next_batch(batch_size):
            train_cost += sess.run([ae['cost'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['train']: True, ae['keep_prob']: 1.0})[0]
            if batch_i % 10 == 0:
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'], feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0})
                m = utils.montage(recon.reshape((-1, 28, 28)),
                    'manifold_%08d.png' % t_i)
                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0})
                m = utils.montage(recon.reshape(
                    (-1, 28, 28)), 'reconstruction_%08d.png' % t_i)
                t_i += 1
            batch_i += 1

        valid_i = 0
        valid_cost = 0
        for batch_xs, _ in mnist.valid.next_batch(batch_size):
            valid_cost += sess.run([ae['cost']], feed_dict={
                ae['x']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0})[0]
        print('train:', train_cost / train_i, 'valid:', valid_cost / valid_i)


def test_celeb():
    """Train an autoencoder on Celeb Net.
    """
    files = CELEB()
    train_vae(
        files=files,
        input_shape=[218, 178, 3],
        batch_size=100,
        n_epochs=50,
        crop_shape=[64, 64, 3],
        crop_factor=0.8,
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='celeb.ckpt')


def test_sita():
    """Train an autoencoder on Sita Sings The Blues.
    """
    if not os.path.exists('sita'):
        os.system('wget http://ossguy.com/sita/Sita_Sings_the_Blues_640x360_XviD.avi')
        os.mkdir('sita')
        os.system('ffmpeg -i Sita_Sings_the_Blues_640x360_XviD.avi -r 60 -f' +
                  ' image2 -s 160x90 sita/sita-%08d.jpg')
    files = [os.path.join('sita', f) for f in os.listdir('sita')]

    train_vae(
        files=files,
        input_shape=[90, 160, 3],
        batch_size=100,
        n_epochs=50,
        crop_shape=[90, 160, 3],
        crop_factor=1.0,
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='sita.ckpt')

def eval_local(files,
              input_shape,
              batch_size=64,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=1,
              save_step=5000,
              ckpt_name="vae.ckpt"):
    
    
    batch = input_pipeline_local(['300w-gt-test.txt'], batch_size=batch_size, shape=[128, 128, 1], is_training=False)
    ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    
    # opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("align/")]
    # ori_vars = [v for v in tf.all_variables() if not v.name.startswith("align/")]
    # batch_idx = tf.Variable(0, dtype=tf.int32)
    # learning_rate = tf.train.exponential_decay(0.0, batch_idx * batch_size, 150000, 0.95, staircase=True)
    # optimizer = tf.train.AdamOptimizer(
    #     learning_rate=learning_rate).minimize(ae['cost'], var_list=opt_vars, global_step=batch_idx)

    avg_pred = ae['y']
    gt_truth = ae['label']
    gt_truth = tf.reshape(gt_truth, (-1, 68, 2))
    # Calculate predictions.
    #norm_error = utils.normalized_rmse(avg_pred, gt_truth)
    # We create a session to use the graph
    sess = tf.Session()
    saver = tf.train.Saver()
    # saver_m = tf.train.Saver(tf.all_variables())
    # sess.run(tf.initialize_all_variables())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()


    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):#+'.index'):
        saver.restore(sess, ckpt_name)
        print('load ' + ckpt_name + ' successfully')


    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = 689 
    #test_xs, test_label = sess.run(batch)
    # test_xs = test_xs
    #print(test_xs.max())
    # utils.montage_landmarks(test_label[:8], 'map_train/test_xs.png')
    all_err = []
    fucking_err = []
    predictions = []
    gts = []
    
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, label_xs, locs = sess.run(batch)
            #batch_xs = batch_xs
            train_cost, landMaps = sess.run([ae['cost'], ae['y']], feed_dict={
                ae['x']: batch_xs, ae['label']: label_xs, ae['train']: False,
                ae['keep_prob']: keep_prob})[:2]
            pred = utils.getLocation(landMaps)
            # Directly from map
             
            #flipped = np.asarray([utils.flip_img(batch_xs[0])])
            #import pdb; pdb.set_trace()
            #pred_flip = sess.run(ae['y'], feed_dict={ae['x']:flipped, ae['train']: False, ae['keep_prob']: keep_prob, ae['keep']: False})
            #import pdb; pdb.set_trace()
            #pred_flip = np.asarray([utils.flip_landmarks(pred_flip[0])])
            predictions.append(pred)
            gts.append(locs[0].reshape([68, 2]))
            print(batch_i, train_cost)
            cost += train_cost
            if batch_i % n_files == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:

                locs = locs.reshape([-1, 68, 2])
                #import pdb; pdb.set_trace()
                err = utils.evaluateBatchError(locs, pred, batch_size)
                all_err.append(err)
                #fucking_err.append(norm_err)
                print('Mean error:' + np.array_str(err))
                #print(norm_err)
                t_i += 1


    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        all_err = np.asarray(all_err)
        all_err = all_err.ravel()
        auc_at_08 = (all_err < .1).mean()
        auc_at_05 = (all_err < .05).mean()
        output = open('300-w-vae.pkl', 'wb')
        pickle.dump(all_err, output)
        output.close()

        print('Overall mean error: ' + np.array_str(all_err.mean(axis=0)))
        print('LFPW: %f, HELEN: %f, IBUG: %f' %(all_err[:224].mean(), all_err[224:554].mean(), all_err[554:].mean()))
        print('Fucking error: %f, auc@05: %.4f, auc@08: %.4f.' %(all_err.mean(), auc_at_05, auc_at_08))
        # saver_m.save(sess, "./" + "align",
        #     global_step=batch_i,
        #     write_meta_graph=False)
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

if __name__ == '__main__':
    test_celeb()
