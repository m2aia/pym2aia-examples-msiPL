# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Neural Network Architecture (VAE_BN)

    Keras-based implementation of a fully connected variational autoecnoder
    equipped with Batch normalization to correct for covariate shift and improve learning stability

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda, Input, Dense, ReLU, BatchNormalization, Layer
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model


class VAELossLayer(Layer):
    """Custom layer to compute VAE loss"""
    def __init__(self, n_features, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.n_features = n_features
        
    def call(self, inputs):
        """Inputs: [z_mean, z_log_var, x_true, x_pred]"""
        z_mean, z_log_var, x_true, x_pred = inputs
        
        # KL divergence
        kl_loss = 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(x_true - x_pred), axis=-1)
        reconstruction_loss *= self.n_features
        
        # Total loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        
        return x_pred


class VAE_BN(object):
    
    def __init__ (self, nSpecFeatures,  intermediate_dim, latent_dim):
        self.nSpecFeatures = nSpecFeatures
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        
    @tf.function
    def sampling(self, args):
        """
        Reparameterization trick by sampling from a continuous function (Gaussian with an auxiliary variable ~N(0,1)).
        [see Our methods and for more details see arXiv:1312.6114]
        """
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim)) # random_normal (mean=0 and std=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

    def get_architecture(self):
        # =========== 1. Encoder Model================
        input_shape = (self.nSpecFeatures, )
        inputs = Input(shape=input_shape, name='encoder_input')
        h = Dense(self.intermediate_dim)(inputs)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        z_mean = Dense(self.latent_dim, name = 'z_mean')(h)
        z_mean = BatchNormalization()(z_mean)
        z_log_var = Dense(self.latent_dim, name = 'z_log_var')(h)
        z_log_var = BatchNormalization()(z_log_var)
        
        # Reparametrization Tric:
        z = Lambda(self.sampling, output_shape = (self.latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name = 'encoder')
        # print("==== Encoder Architecture...")
        # encoder.summary()
        # plot_model(encoder, to_file='VAE_BN_encoder.png', show_shapes=True)
        
        # =========== 2. Encoder Model================
        latent_inputs = Input(shape = (self.latent_dim,), name='Latent_Space')
        hdec = Dense(self.intermediate_dim)(latent_inputs)
        hdec = BatchNormalization()(hdec)
        hdec = ReLU()(hdec)
        outputs = Dense(self.nSpecFeatures)(hdec)
        self.decoder = Model(latent_inputs, outputs, name = 'decoder')
        # print("==== Decoder Architecture...")
        # self.decoder.summary()       
        # plot_model(decoder, to_file='VAE_BN__decoder.png', show_shapes=True)
        
        #=========== VAE_BN: Encoder_Decoder ================
        encoder_outputs = encoder(inputs)
        z_mean_out, z_log_var_out, z = encoder_outputs
        outputs = self.decoder(z)
        
        # Add VAE loss computation
        outputs_with_loss = VAELossLayer(self.nSpecFeatures)([z_mean_out, z_log_var_out, inputs, outputs])
        
        VAE_BN_model = Model(inputs, outputs_with_loss, name='VAE_BN')
        
        # ========== Compile VAE_BN model ===========
        VAE_BN_model.compile(optimizer='adam')
        return VAE_BN_model, encoder






