import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class Autoencoder(Model):
  def __init__(self, input_dim, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(input_dim,)),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(input_dim, activation='sigmoid'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Autoencoder1(Model):
  def __init__(self, input_dim, latent_dim):
    super(Autoencoder1, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(input_dim,)),
      layers.Dense(input_dim//10, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(input_dim//10, activation='relu'),
      layers.Dense(input_dim, activation='sigmoid'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Convolutional_Autoencoder(Model):
  def __init__(self, input_dim):
    super(Convolutional_Autoencoder, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(input_dim[0],input_dim[1],1)),
      layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=3, strides=2, activation='relu', padding='same')])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(3, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=2, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Convolutional_Autoencoder1(Model):
  def __init__(self, input_dim, latent_dim):
    super(Convolutional_Autoencoder1, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(input_dim[0],input_dim[1],1)),
      layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid')])

    self.decoder = tf.keras.Sequential([
      layers.Dense(7*7*16, activation='relu'),
      layers.Reshape((7,7,16)),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=2, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Convolutional_Autoencoder2(Model):
  def __init__(self, latent_dim):
    super(Convolutional_Autoencoder2, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(125, 125, 3)),
      layers.Conv2D(64, kernel_size=3, strides=3, activation='relu', padding='same'),
      layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='valid'),
      layers.Conv2D(30, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(10, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid')])

    self.decoder = tf.keras.Sequential([
      layers.Dense(5*5*10, activation='relu'),
      layers.Reshape((5, 5, 30)),
      layers.Conv2DTranspose(10, kernel_size=3, strides=3, activation='relu', padding='valid'),# after 14
      layers.Conv2DTranspose(30, kernel_size=3, strides=2, activation='relu', padding='valid'),# after: 20
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),# after: 40
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='valid'),# after : 120
      layers.Conv2D(3, kernel_size=2, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Convolutional_Autoencoder_gamma_e(Model):
  def __init__(self, latent_dim):
    super(Convolutional_Autoencoder_gamma_e, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(32, 32, 1)),
      layers.Conv2D(8, kernel_size=4, strides=1, activation='relu', padding='same'),
      layers.Conv2D(16, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='same'),
      layers.Conv2D(64, kernel_size=2, strides=2, activation='relu', padding='same'),#,
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid')])

    self.decoder = tf.keras.Sequential([
      layers.Dense(4*4*64, activation='relu'),
      layers.Reshape((4, 4, 64)),
      layers.Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='same'),# after 14
      layers.Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding='same'),# after: 20
      layers.Conv2DTranspose(16, kernel_size=4, strides=2, activation='relu', padding='same'),# after: 40
      layers.Conv2DTranspose(8, kernel_size=4, strides=1, activation='relu', padding='same'),# after : 120
      layers.Conv2D(1, kernel_size=2, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Convolutional_Autoencoder_hp_model(Model):
  def __init__(self, latent_dim, filters, kernel_sizes):
    super(Convolutional_Autoencoder_hp_model, self).__init__()

    enc = [layers.Input(shape=(32, 32, 1))]

    enc.append(layers.Conv2D(filters[0][0], kernel_size=4, strides=1, activation='relu', padding='same'))
    if len(filters[0]) > 1:
        for i,j in zip(filters[0][1:], kernel_sizes[0]):
          enc.append(layers.Conv2D(i, kernel_size=j, strides=1, activation='relu', padding='same'))

    enc.append(layers.Conv2D(filters[1][0], kernel_size=4, strides=2, activation='relu', padding='same'))
    if len(filters[1]) > 1:
        for i,j in zip(filters[1][1:], kernel_sizes[1]):
          enc.append(layers.Conv2D(i, kernel_size=j, strides=1, activation='relu', padding='same'))

    enc.append(layers.Conv2D(filters[2][0], kernel_size=4, strides=2, activation='relu', padding='same'))
    if len(filters[2]) > 1:
        for i,j in zip(filters[2][1:], kernel_sizes[2]):
          enc.append(layers.Conv2D(i, kernel_size=j, strides=1, activation='relu', padding='same'))

    enc.append(layers.Conv2D(filters[3][0], kernel_size=2, strides=2, activation='relu', padding='same'))
    if len(filters[3]) > 1:
        for i,j in zip(filters[3][1:], kernel_sizes[3]):
          enc.append(layers.Conv2D(i, kernel_size=j, strides=1, activation='relu', padding='same'))

    enc.append(layers.Flatten())
    enc.append(layers.Dense(latent_dim, activation='sigmoid'))

    self.encoder = tf.keras.Sequential(enc)

    dec = []

    dec.append(layers.Dense(4*4*filters[3][-1], activation='relu'))
    dec.append(layers.Reshape((4, 4, filters[3][-1])))

    if len(filters[3]) > 1:
        for i,j in reversed(list(zip(filters[3][1:], kernel_sizes[3]))):
          dec.append(layers.Conv2DTranspose(i, kernel_size=j, strides=1, activation='relu', padding='same'))
    dec.append(layers.Conv2DTranspose(filters[3][0], kernel_size=2, strides=2, activation='relu', padding='same'))

    if len(filters[2]) > 1:
        for i,j in reversed(list(zip(filters[2][1:], kernel_sizes[2]))):
          dec.append(layers.Conv2DTranspose(i, kernel_size=j, strides=1, activation='relu', padding='same'))
    dec.append(layers.Conv2DTranspose(filters[2][0], kernel_size=4, strides=2, activation='relu', padding='same'))

    if len(filters[1]) > 1:
        for i,j in reversed(list(zip(filters[1][1:], kernel_sizes[1]))):
          dec.append(layers.Conv2DTranspose(i, kernel_size=j, strides=1, activation='relu', padding='same'))
    dec.append(layers.Conv2DTranspose(filters[1][0], kernel_size=4, strides=2, activation='relu', padding='same'))

    if len(filters[0]) > 1:
        for i,j in reversed(list(zip(filters[0][1:], kernel_sizes[0]))):
          dec.append(layers.Conv2DTranspose(i, kernel_size=j, strides=1, activation='relu', padding='same'))
    dec.append(layers.Conv2DTranspose(filters[0][0], kernel_size=4, strides=1, activation='relu', padding='same'))

    dec.append(layers.Conv2D(1, kernel_size=2, activation='sigmoid', padding='same'))

    self.decoder = tf.keras.Sequential(dec)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class QG_Convolutional_Autoencoder(Model):
  def __init__(self, latent_dim):
    super(QG_Convolutional_Autoencoder, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(125, 125, 3)),
      layers.Conv2D(16, kernel_size=6, strides=1, activation='relu', padding='same'),
      layers.Conv2D(16, kernel_size=6, strides=1, activation='relu', padding='valid'),
      layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.Conv2D(30, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Flatten(),
      layers.Dense(latent_dim//0.75, activation='relu'),
      layers.Dense(latent_dim, activation='sigmoid')])

    self.decoder = tf.keras.Sequential([
      layers.Dense(latent_dim//0.75, activation='relu'),
      layers.Dense(15*15*30, activation='relu'),
      layers.Reshape((15, 15, 30)),
      layers.Conv2DTranspose(30, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=6, strides=1, activation='relu', padding='valid'),
      layers.Conv2DTranspose(16, kernel_size=6, strides=1, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

        latent_dim = 15

        encoder_inputs = keras.Input(shape=(32, 32, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((8, 8, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")


        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x):
      z_mean, z_log_var, z = self.encoder(x)
      reconstruction = self.decoder(z)
      return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
