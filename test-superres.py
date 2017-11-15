#!/usr/bin/env python3

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

parser = argparse.ArgumentParser()
parser.add_argument("image_file", type=str, help="Path to an image file to upscale.")
parser.add_argument("out_image_file", type=str, help="Path to output the upscaled image file.")
args = parser.parse_args()

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
	print("Weights not found.")
	sys.exit(1)

image_file_path = Path(args.image_file)
if not image_file_path.exists():
	print("File does not exist.")
	sys.exit(2)

# Build model
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

# Multi-scale Reconstruction
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

def lerp(x, from_min, From_max, to_min, to_max):
	return (x / b) * a

def upscale(orig_img):
	figure = np.zeros((orig_img.size[1] * 2, orig_img.size[0] * 2, 3))
	new_size = (orig_img.size[0] * 2, orig_img.size[1] * 2)
	
	blend_weights = []
	for offset in [0, 32]: # this makes the blending take place after the initial prediction
		for y in range(0 + offset, new_size[1], 64):
			for x in range(0 + offset, new_size[0], 64):
				img_x = orig_img.crop((int(x/2), int(y/2), int(x/2) + 32, int(y/2) + 32))
				img_x = img_x.resize(supersampler_input_shape[:2])
				# print(img_x.size)
				array_x = img_to_array(img_x)
				max_pixel_value = array_x.max()
				array_x /= max_value

				print("predicting at ", x, ",", y)
				chunks_out = supersampler.predict(np.array([array_x]))
				img_y = array_to_img(chunks_out[0])
				if x + 64 > new_size[0]:
					# print("x too big - ", x + 64, " > ", new_size[0], " | img_y size:", img_y.size)
					img_y = img_y.crop((0, 0, new_size[0]-x, img_y.size[0]))
					# print("new img_y size:", img_y.size)
				if y + 64 > new_size[1]:
					# print("y too big - img_y size:", img_y.size)
					img_y = img_y.crop((0, 0, img_y.size[1], new_size[1]-y))
				# print(x, y, figure.shape, img_y.size, orig_img.size)
				try:
					array_y = img_to_array(img_y)
					array_y *= max_pixel_value
					# print(type(array_y))

					# image math: blending - https://homepages.inf.ed.ac.uk/rbf/HIPR2/blend.htm
					# https://stackoverflow.com/questions/5919663/how-does-photoshop-blend-two-images-together
					if len(blend_weights) == 0:
						blend_weights = np.array([np.concatenate([np.linspace(0, 1, int(array_y.shape[1] / 2)), np.linspace(1, 0, int(array_y.shape[1] / 2))]) for _ in range(array_y.shape[0])])
						blend_weights = np.array([blend_weights for _ in range(3)])
						blend_weights = blend_weights.reshape(array_y.shape)
					if (offset == 32 and x % 32 == 0 and x % 64 != 0) or (offset == 32 and y % 32 == 0 and y % 64 != 0):
						# print("blending X")
						# print("blend_weights:", blend_weights.shape)
						source = figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] * (1 - blend_weights)
						target = array_y * blend_weights
						result = source + target
						figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] = result
					if offset == 0:
						# print("applying")
						figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] = array_y
						
				except ValueError as e:
					print(x, y, figure.shape, img_y.size, "ValueError")
					# raise e
					pass
			
	return array_to_img(figure)

print("upscaling")
orig_img = load_img(image_file_path)
upscaled = upscale(orig_img)
upscaled.save(args.out_image_file)
print("Image saved to", args.out_image_file)