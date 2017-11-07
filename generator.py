#!/usr/bin/env python3

import h5py, argparse, sys, os, pandas
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Layer, Dropout
from keras.layers import Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras import metrics
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

ckpt_path = Path("./ckpt/")
ckpt_file = ckpt_path / 'vae_weights.h5'

datagen = ImageDataGenerator(
			rotation_range=0,
			width_shift_range=0,
			height_shift_range=0,
			rescale=1./255,
			shear_range=0.1,
			zoom_range=0,
			horizontal_flip=False,
			fill_mode='nearest',
			data_format="channels_last")

# NOTE: do not use odd numbered dimentions or odd multiples of the ratio 16:9
img_rows, img_cols, img_chns = 46, 80, 3
# img_rows, img_cols, img_chns = 54, 96, 3 # not valid
# img_rows, img_cols, img_chns = 72, 128, 3
# img_rows, img_cols, img_chns = 90, 160, 3
# img_rows, img_cols, img_chns = 180, 320, 3
# img_rows, img_cols, img_chns = 360, 640, 3

if K.image_data_format() == 'channels_first':
	image_size = (img_chns, img_rows, img_cols)
else:
	image_size = (img_rows, img_cols, img_chns)

batch_size = 10
original_dim = image_size[0] * image_size[1] * image_size[2]
intermediate_dim = 4000
# intermediate_dim = int(original_dim / 4)
# latent_dim = 512
latent_dim = 16
epsilon_std = 1.0
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3


x = Input(shape=image_size)
conv_1 = Conv2D(img_chns,
				kernel_size=(2, 2),
				padding='same', activation='relu',
				name="conv_1")(x)
conv_2 = Conv2D(filters,
				kernel_size=(2, 2),
				padding='same', activation='relu',
				strides=(2, 2),
				name="conv_2")(conv_1)
conv_3 = Conv2D(filters,
				kernel_size=num_conv,
				padding='same', activation='relu',
				strides=1,
				name="conv_3")(conv_2)
conv_4 = Conv2D(filters,
				kernel_size=num_conv,
				padding='same', activation='relu',
				strides=1,
				name="conv_4")(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu', name="hidden")(flat)
z_mean = Dense(latent_dim, name="z_mean")(hidden)
z_log_var = Dense(latent_dim, name="z_log_var")(hidden)


def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
							  stddev=epsilon_std)
	return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu', name="decoder_hid")
decoder_upsample = Dense(filters * int(img_rows / 2) * int(img_cols / 2), activation='relu', name="decoder_upsample")

if K.image_data_format() == 'channels_first':
	output_shape = (batch_size, filters, int(img_rows / 2), int(img_cols / 2))
else:
	output_shape = (batch_size, int(img_rows / 2), int(img_cols / 2), filters)

decoder_reshape = Reshape(output_shape[1:], name="decoder_reshape")
decoder_deconv_1 = Conv2DTranspose(filters,
								   kernel_size=num_conv,
								   padding='same',
								   strides=1,
								   activation='relu',
								   name="decoder_deconv_1")
decoder_deconv_2 = Conv2DTranspose(filters,
								   kernel_size=num_conv,
								   padding='same',
								   strides=1,
								   activation='relu',
								   name="decoder_deconv_2")

if K.image_data_format() == 'channels_first':
	output_shape = (batch_size, filters, int(img_rows / 2) + 1, int(img_cols / 2) + 1)
else:
	output_shape = (batch_size, int(img_rows / 2) + 1, int(img_cols / 2) + 1, filters)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
										  kernel_size=(3, 3),
										  strides=(2, 2),
										  padding='valid',
										  activation='relu',
										  name="decoder_deconv_3_upsamp")
decoder_mean_squash = Conv2D(img_chns,
							 kernel_size=2,
							 padding='valid',
							 activation='sigmoid',
							 name="decoder_mean_squash")

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
class CustomVariationalLayer(Layer):
	"""
	We use this custom layer to implement VAE scoring
	"""

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(CustomVariationalLayer, self).__init__(**kwargs)

	def vae_loss(self, x, x_decoded_mean_squash):
		x = K.flatten(x)
		x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
		xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
		kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		x = inputs[0]
		x_decoded_mean_squash = inputs[1]
		loss = self.vae_loss(x, x_decoded_mean_squash)
		self.add_loss(loss, inputs=inputs)
		# We don't use this output.
		return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

if ckpt_file.exists():
	print("loading model from checkpoint")
	vae.load_weights(str(ckpt_file))


# test replication of input
print("loading images")
tags_file = Path("data/good/tags.csv")
tags = pandas.read_csv(tags_file, names=["file", "tagA", "tagB", "tagC", "tagD", "tagE"])
cars = tags.query("tagA == 'car'")
cars_images = [img_to_array(load_img(str(Path("data/good/img") / x)).resize((img_cols, img_rows))) for x in cars["file"]]
image_batch = np.array(cars_images)
image_batch *= (1/image_batch.max())
n = 8
figure = np.zeros((image_size[0] * n, image_size[1] * 2, img_chns))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, 2))

x_decoded = vae.predict(image_batch[:n])
for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		if j == 0:
			out = image_batch[i].reshape(image_size)
		else:
			out = x_decoded[i].reshape(image_size)
		figure[i * image_size[0]: (i + 1) * image_size[0],
			   j * image_size[1]: (j + 1) * image_size[1]] = out
plt.figure(figsize=(30, 18))
plt.imshow(figure)
plt.show()

# test new sample generation
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,), name="decoder_input")
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

print("generating small random samples")
# display a 2D manifold
n = 2  # number of image rows and columns images
out_size = image_size
figure = np.zeros((image_size[0] * n, image_size[1] * n, img_chns))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.random.random_sample(size=(n*n, latent_dim,))
		x_decoded = generator.predict(z_sample)
		out = x_decoded[i+j].reshape(out_size)
		figure[i * image_size[0]: (i + 1) * image_size[0],
			   j * image_size[1]: (j + 1) * image_size[1]] = out

plt.figure(figsize=(30, 18))
plt.imshow(figure)
plt.show()
