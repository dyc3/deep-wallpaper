#!/usr/bin/env python3
# coding: utf-8

import h5py, argparse, sys, os, pandas
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Layer, Dropout
from keras.layers import Flatten, Activation
from keras.layers import concatenate, add
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras import metrics
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# # Image Super Resolution
#
# The objective here is to do a realistic super resolution by 2 on any given image. To save memory, we will do super resolution in chunks of 32x32 => 64x64.

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

ckpt_path = Path("./ckpt/")
ckpt_file = ckpt_path / 'superres_weights.h5'
if not ckpt_path.exists():
	ckpt_path.mkdir(parents=True)

# ## Build Model
#
# References:
# * https://mtyka.github.io/machine/learning/2017/08/09/highres-gan-faces-followup.html
# * https://arxiv.org/pdf/1612.03242.pdf
# * https://arxiv.org/pdf/1607.07680.pdf (https://github.com/MarkPrecursor/EEDS-keras)


supersampler_input_shape = (32, 32, 3)
supersampler_output_shape = (64, 64, 3)
super_out_dim = reduce(lambda x, y: x * y, supersampler_output_shape, 1)

# super_input_latent = Input(shape=(latent_dim,),
# 						   name="supersampler_input_latent")
super_input_decoded = Input(shape=supersampler_input_shape,
							name="supersampler_input_decoded")

def Res_block():
	_input = Input(shape=(None, None, 64))

	conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
	conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(conv)

	out = add(inputs=[_input, conv])
	out = Activation('relu')(out)

	model = Model(inputs=_input, outputs=out)

	return model

# _input = Input(shape=(None, None, 1), name='input')
_input = Input(shape=supersampler_input_shape, name='input')

Feature = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
Feature_out = Res_block()(Feature)

# Upsampling
Upsampling1 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
Upsampling2 = Conv2DTranspose(filters=4, kernel_size=(14, 14), strides=(2, 2),
							  padding='same', activation='relu')(Upsampling1)
Upsampling3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

# Mulyi-scale Reconstruction
Reslayer1 = Res_block()(Upsampling3)

Reslayer2 = Res_block()(Reslayer1)

# ***************//
Multi_scale1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Reslayer2)

Multi_scale2a = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale1)

Multi_scale2b = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale1)
Multi_scale2b = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale2b)

Multi_scale2c = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale1)
Multi_scale2c = Conv2D(filters=16, kernel_size=(5, 1), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale2c)

Multi_scale2d = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale1)
Multi_scale2d = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1),
					   padding='same', activation='relu')(Multi_scale2d)

Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

out = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)
supersampler = Model(inputs=_input, outputs=out)
supersampler.compile(optimizer="rmsprop", loss='mean_squared_error')

supersampler.summary()


if ckpt_file.exists():
	print("loading model from checkpoint")
	supersampler.load_weights(str(ckpt_file))


# ## Generate Training Data from Images
#
# We'll grab images, resize them by half, and cut corresponding chunks out of both the original and the resized images.


img_dir = Path("data/good/img")
img_path_gen = img_dir.iterdir()

imgs_processed = []
dataX = []
dataY = []

sm_chunk_size = (32, 32)

images_to_process = 64 # just for testing purposes
step = sm_chunk_size[0]

def getSmallImageSize(orig_size):
	assert len(orig_size) == 2

	small_size = list(orig_size)
	for i in range(len(small_size)):
		small_size[i] = int(small_size[i] / 2)
	return tuple(small_size)

for i in range(images_to_process):
	img_path = next(img_path_gen)
	img_orig = load_img(img_path)
	# print("orig: ", img_orig)
	small_size = getSmallImageSize(img_orig.size)
	img_small = img_orig.copy().resize(small_size)
	# print("small:", img_small)

	chunks = 0
	for y in range(0, small_size[1] - sm_chunk_size[1], step):
		for x in range(0, small_size[0] - sm_chunk_size[0], step):
			img_x = img_small.crop((x, y, x + 32, y + 32))
			img_y = img_orig.crop((x*2, y*2, x*2 + 64, y*2 + 64))

			img_x = img_x.resize(supersampler_input_shape[:2])
			img_y = img_y.resize(supersampler_output_shape[:2])

			dataX.append(img_to_array(img_x)) # maybe / 255 needed?
			dataY.append(img_to_array(img_y)) # maybe / 255 needed?

			chunks += 1
	imgs_processed.append((img_path, chunks))


dataX = np.array(dataX)
dataY = np.array(dataY)
dataX *= (1/255)
dataY *= (1/255)
print(dataX.shape)
print(dataY.shape)


# ## Train the model
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(monitor='loss', filepath=str(ckpt_file), save_best_only=True, verbose=1)

supersampler.fit(dataX, dataY, batch_size=128, epochs=4000, verbose=1, callbacks=[earlystop, checkpointer])

sys.exit(0)
# ## Test the model
# ### Some raw visual comparisons

# Compare the 32x32 with the predicted 64x64 and real 64x64


start = 10
count = 30

pred_x = dataX[start:start+count] # the chunks to predict
real_y = dataY[start:start+count]
pred_y = supersampler.predict(pred_x)

print(imgs_processed[start])
for i in range(count):
	plt.imshow(pred_x[i])
	plt.show()
	plt.imshow(pred_y[i])
	plt.show()
	plt.imshow(real_y[i])
	plt.show()


# In[97]:


for i in range(0, 30, 1):
	print(imgs_processed[i])
	orig_img = load_img(imgs_processed[i][0])

	figure = np.zeros((orig_img.size[1], orig_img.size[0], 3))
	small_img = orig_img.resize(getSmallImageSize(orig_img.size))
	plt.figure(figsize=(12,6))
	plt.title("Low Res Image")
	plt.imshow(small_img)
	plt.show()

	for y in range(0, orig_img.size[1], 64):
		for x in range(0, orig_img.size[0], 64):
			img_x = small_img.crop((int(x/2), int(y/2), int(x/2) + 32, int(y/2) + 32))
			img_x = img_x.resize(supersampler_input_shape[:2])
			array_x = img_to_array(img_x)
			array_x /= 255

			chunks_out = supersampler.predict(np.array([array_x]))
			img_y = array_to_img(chunks_out[0])
			if x + 64 > orig_img.size[0]:
				# print("x too big - img_y size:", img_y.size)
				img_y = img_y.crop((0, 0, orig_img.size[0]-x, img_y.size[0]))
			if y + 64 > orig_img.size[1]:
				# print("y too big - img_y size:", img_y.size)
				img_y = img_y.crop((0, 0, img_y.size[1], orig_img.size[1]-y))
			# print(x, y, figure.shape, img_y.size, orig_img.size)
			try:
				array_y = img_to_array(img_y)
				array_y *= 255
				# print(type(array_y))
				figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] = array_y
			except ValueError:
				print(x, y, figure.shape, img_y.size, "ValueError")
				pass

	plt.figure(figsize=(20,10))
	plt.title("Predicted Image")
	plt.imshow(figure)
	plt.show()

	plt.figure(figsize=(20,10))
	plt.title("Original Image")
	plt.imshow(orig_img)
	plt.show()
