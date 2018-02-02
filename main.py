#!/usr/bin/env python3

import h5py, argparse, sys, os
from PIL import Image
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict
import glob, re
from functools import reduce
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dropout
from keras.layers import Flatten, Activation, Embedding
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import concatenate, multiply, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.utils.generic_utils import Progbar
from keras import activations
from vis.visualization import visualize_activation
from vis.utils import utils

# NOTE: These are not good tags. Good tags would probably consist of art styles instead of objects found in the photo.
# NOTE: The order of these elements matters. Do not change the order.
tags = ["car","minimal","nature","animal","landscape","people","abstract","city","watermark","text","space","sci-fi","interior","fantasy"]

parser = argparse.ArgumentParser()
parser.add_argument("--show-summary", help="Prints summaries of the model's architecture.", action="store_true")
# parser.add_argument("--train", action="store_true")
parser.add_argument("--train", type=str, choices=["acgan", "supersampler"])
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--resume", type=int, default=1, help="The epoch at which to resume training. (Will load from the previous epoch)")
parser.add_argument("--steps-per-epoch", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=32)

parser.add_argument("--data-dir", type=str, default="data/good/img")
parser.add_argument("--tags-file", type=str, default="data/good/tags.csv")

parser.add_argument("--export-full-model", type=str)
parser.add_argument("--export-model-json", type=str)

parser.add_argument("--generate", type=int, default=0)
parser.add_argument("--gen-upscale", type=int, default=3, help="The number of times to upscale generated images.")
parser.add_argument("--upscale", type=str)
parser.add_argument("--tags", nargs="+", help="Specify tags to use when generating images, otherwise random tags will be used.", default=[])

parser.add_argument("--visualize", type=str, choices=["epochs", "layers", "model"])
parser.add_argument("--evaluate", type=str, help="Evaluate all epochs and put into specified file, formatted as csv.")
args = parser.parse_args()

img_width, img_height, img_chns = 128, 72, 3
tags_file = Path("data/good/tags.csv")

visualization_dir = Path("visualization/")
if not visualization_dir.exists():
	visualization_dir.mkdir(parents=True)

ckpt_dir = Path("ckpt/")
if not ckpt_dir.exists():
	ckpt_dir.mkdir(parents=True)

gen_dir = Path("gen/")
if not gen_dir.exists():
	gen_dir.mkdir(parents=True)

class ACGAN(object):
	"""Handles training and image generation for the ACGAN.
	"""

	# NOTE: 100 latent space is too small. If you retrain this, make sure to have a lot more latent space.
	def __init__(self, latent_size=100, tags=[]):
		assert latent_size > 0
		assert len(tags) > 0

		self.latent_size = latent_size
		self.tags = tags

		self.generator = self.build_generator()
		self.discriminator = self.build_discriminator()
		self.combined = self.build_combined()

	@property
	def num_classes(self):
		return len(self.tags)

	def get_tags_from_file(self, img_path):
		with open(str(tags_file), "r") as f:
			line = f.readline()
			while line:
				split = line.rstrip("\n").split(",")
				if split[0] == img_path.name:
					return [x for x in split[1:] if x in tags]
				line = f.readline()
		return []

	def tags_to_index(self, in_tags):
		indices = []
		for tag in in_tags:
			if tag in tags:
				indices.append(tags.index(tag))
		return np.array(indices)

	def tags_to_embeddings(self, img_tags):
		embeddings = []
		for tag in tags:
			embeddings.append(1 if tag in img_tags else 0)
		return np.array(embeddings, dtype="float32")

	def random_tags(self):
		return np.random.choice(self.tags, size=np.random.randint(1, len(self.tags)))

	def batch_generator(self):
		data_dir = Path(args.data_dir)
		all_paths = np.asarray(list(data_dir.iterdir()))
		np.random.shuffle(all_paths)
		batch_start = 0
		# generate batches forever
		while True:
			batch = []
			checked = 0
			while len(batch) < args.batch_size:
				img_path = all_paths[(batch_start + checked) % (len(all_paths) - 1)]
				img_tags = self.get_tags_from_file(img_path)
				checked += 1
				if len(img_tags) == 0:
					continue
				img_tags = self.tags_to_embeddings(img_tags)
				img = img_to_array(load_img(img_path).resize((img_width, img_height)))
				img /= 127.5
				img -= 1
				batch.append((img, img_tags))

			batch_start += checked
			if batch_start >= len(all_paths):
				batch_start = 0

			yield np.asarray(batch)

	def build_generator(self):
		assert self.latent_size > 0
		assert self.num_classes > 0
		# we will map a pair of (z, L), where z is a latent vector and L is a
		# label drawn from P_c, to image space (..., 28, 28, 1)
		cnn = Sequential()

		cnn.add(Dense((self.latent_size + self.num_classes), input_dim=(self.latent_size + self.num_classes), activation='relu'))
		cnn.add(Dense(128 * 16 * 9 * img_chns, activation='relu'))
		cnn.add(Reshape((9, 16, 128 * img_chns)))

		# upsample to (18, 32, ...)
		cnn.add(UpSampling2D(size=(2, 2)))
		cnn.add(Conv2D(256, 5, padding='same',
					activation='relu',
					kernel_initializer='glorot_normal'))

		# upsample to (36, 64, ...)
		cnn.add(UpSampling2D(size=(2, 2)))
		cnn.add(Conv2D(128, 5, padding='same',
					activation='relu',
					kernel_initializer='glorot_normal'))

		# upsample to (72, 128, ...)
		cnn.add(UpSampling2D(size=(2, 2)))
		cnn.add(Conv2D(128, 5, padding='same',
					activation='relu',
					kernel_initializer='glorot_normal'))

		# take a channel axis reduction
		cnn.add(Conv2D(3, 2, padding='same',
					activation='tanh',
					kernel_initializer='glorot_normal'))
		
		if args.show_summary:
			cnn.summary()

		# this is the z space commonly refered to in GAN papers
		latent = Input(shape=(self.latent_size, ), dtype="float32", name="latent")

		# this will be our label
		image_class = Input(shape=(self.num_classes,), dtype="float32", name="tags")
		cls = Dense(self.num_classes)(image_class)
		
		# hadamard product between z-space and a class conditional embedding
		h = concatenate([latent, cls])

		fake_image = cnn(h)

		return Model([latent, image_class], fake_image, name="generator")

	def build_discriminator(self):
		assert self.num_classes > 0
		# build a relatively standard conv net, with LeakyReLUs as suggested in
		# the reference paper
		cnn = Sequential()

		cnn.add(Conv2D(32, 3, padding='same', strides=2,
					input_shape=(img_height, img_width, img_chns)))
		cnn.add(LeakyReLU())
		cnn.add(Dropout(0.3))

		cnn.add(Conv2D(64, 3, padding='same', strides=1))
		cnn.add(LeakyReLU())
		cnn.add(Dropout(0.3))

		cnn.add(Conv2D(128, 3, padding='same', strides=2))
		cnn.add(LeakyReLU())
		cnn.add(Dropout(0.3))

		cnn.add(Conv2D(256, 3, padding='same', strides=1))
		cnn.add(LeakyReLU())
		cnn.add(Dropout(0.3))

		cnn.add(Flatten())

		image = Input(shape=(img_height, img_width, img_chns))

		features = cnn(image)

		# first output (name=generation) is whether or not the discriminator
		# thinks the image that is being shown is fake, and the second output
		# (name=auxiliary) is the class that the discriminator thinks the image
		# belongs to.
		fake = Dense(1, activation='sigmoid', name='generation')(features)
		aux = Dense(self.num_classes, activation='softmax', name='auxiliary')(features)

		return Model(image, [fake, aux], name="discriminator")

	def build_combined(self):
		assert self.latent_size > 0
		assert self.num_classes > 0

		# Adam parameters suggested in https://arxiv.org/abs/1511.06434
		adam_lr = 0.0002
		adam_beta_1 = 0.5

		self.discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'categorical_crossentropy'])

		latent = Input(shape=(self.latent_size, ), name="latent")
		image_class = Input(shape=(self.num_classes,), dtype='float32', name="tags")

		# get a fake image
		fake = self.generator([latent, image_class])

		# we only want to be able to train generation for the combined model
		self.discriminator.trainable = False
		fake, aux = self.discriminator(fake)
		combined = Model([latent, image_class], [fake, aux])

		combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'categorical_crossentropy'])
		if args.show_summary:
			print('Combined model:')
			combined.summary()

		return combined

	def _get_file_name_suffix(self, epoch):
		if not self.latent_size == 100:
			return "latent_{}".format(self.latent_size) + "_epoch_{0:04d}".format(epoch)
		else:
			return "epoch_{0:04d}".format(epoch)

	def load_checkpoint(self, epoch):
		assert isinstance(epoch, int) and epoch > 0
		self.generator.load_weights(str(ckpt_dir / 'params_generator_{}.hdf5'.format(self._get_file_name_suffix(epoch))), True)
		try:
			self.discriminator.load_weights(str(ckpt_dir / 'params_discriminator_{}.hdf5'.format(self._get_file_name_suffix(epoch))), True)
		except OSError:
			print("WARNING: discriminator checkpoint not found.")

	def train(self, epochs, resume=1, batch_size=32):
		if resume > 1:
			# We need to load weights from the last epoch
			self.load_checkpoint(resume - 1)

		batch_gen = self.batch_generator()
		train_history = defaultdict(list)
		test_history = defaultdict(list)

		for epoch in range(resume, epochs + 1):
			progress_bar = Progbar(target=args.steps_per_epoch * batch_size)

			epoch_gen_loss = []
			epoch_disc_loss = []

			for index in range(args.steps_per_epoch):
				# get a batch of real images
				this_batch = next(batch_gen)
				this_batch_size = len(this_batch)
				image_batch, label_batch = zip(*this_batch)
				image_batch = np.array(image_batch)
				label_batch = np.array(label_batch)

				# generate a new batch of noise
				noise = np.random.uniform(-1, 1, (batch_size, self.latent_size))

				# sample some labels from p_c
				sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(batch_size)])

				# generate a batch of fake images, using the generated labels as a
				# conditioner. We reshape the sampled labels to be
				# (batch_size, 1) so that we can feed them into the embedding
				# layer as a length one sequence
				generated_images = self.generator.predict([noise, sampled_labels], verbose=0)

				x = np.concatenate((image_batch, generated_images))

				# use soft real/fake labels
				soft_zero, soft_one = 0.25, 0.75
				y = np.array([soft_one] * this_batch_size + [soft_zero] * batch_size)
				y = y.reshape(-1, 1)
				aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

				# see if the discriminator can figure itself out...
				epoch_disc_loss.append(self.discriminator.train_on_batch(x, {"generation":y, "auxiliary":aux_y}))

				# make new noise. we generate batch_size + this_batch_size here such that we have
				# the generator optimize over an identical number of images as the
				# discriminator
				noise = np.random.uniform(-1, 1, (batch_size + this_batch_size, self.latent_size))
				sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(batch_size + this_batch_size)])

				# we want to train the generator to trick the discriminator
				# For the generator, we want all the {fake, not-fake} labels to say
				# not-fake
				trick = np.ones(batch_size + this_batch_size) * soft_one

				epoch_gen_loss.append(self.combined.train_on_batch(
					[noise, sampled_labels],
					[trick, sampled_labels]))

				progress_bar.update(index * batch_size)

			print('Testing for epoch {}:'.format(epoch))

			# evaluate the testing loss here

			# generate a new batch of noise
			num_test = batch_size
			x_test, y_test = zip(*next(batch_gen))
			noise = np.random.uniform(-1, 1, (num_test, self.latent_size))

			# sample some labels from p_c and generate images from them
			sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(batch_size)])
			generated_images = self.generator.predict([noise, sampled_labels], verbose=False)

			x = np.concatenate((x_test, generated_images))
			y = np.array([1] * len(x_test) + [0] * num_test)
			aux_y = np.concatenate((y_test, sampled_labels), axis=0)

			# see if the discriminator can figure itself out...
			discriminator_test_loss = self.discriminator.evaluate(x, [y, aux_y], verbose=False)

			discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

			# make new noise
			noise = np.random.uniform(-1, 1, (2 * num_test, self.latent_size))
			sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(2 * num_test)])

			trick = np.ones(2 * num_test)

			generator_test_loss = self.combined.evaluate([noise, sampled_labels], [trick, sampled_labels], verbose=False)

			generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

			# generate an epoch report on performance
			train_history['generator'].append(generator_train_loss)
			train_history['discriminator'].append(discriminator_train_loss)

			test_history['generator'].append(generator_test_loss)
			test_history['discriminator'].append(discriminator_test_loss)

			print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *self.discriminator.metrics_names))
			print('-' * 65)

			ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
			print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
			print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
			print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
			print(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1]))

			# save weights every epoch
			self.generator.save_weights(str(ckpt_dir / 'params_generator_{}.hdf5'.format(self._get_file_name_suffix(epoch))), True)
			self.discriminator.save_weights(str(ckpt_dir / 'params_discriminator_{}.hdf5'.format(self._get_file_name_suffix(epoch))), True)

		pickle.dump({'train': train_history, 'test': test_history}, open('acgan-history.pkl', 'wb')) # TODO: load histories and append when resuming training

	def evaluate(self, batch_size=32):
		"""
		Evaluate the current weights.

		Returns a tuple of (generator_loss, discriminator_loss)
		"""
		batch_gen = self.batch_generator()		

		# generate a new batch of noise
		num_test = batch_size
		x_test, y_test = zip(*next(batch_gen))
		noise = np.random.uniform(-1, 1, (num_test, self.latent_size))

		# sample some labels from p_c and generate images from them
		sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(batch_size)])
		generated_images = self.generator.predict([noise, sampled_labels], verbose=False)

		x = np.concatenate((x_test, generated_images))
		y = np.array([1] * len(x_test) + [0] * num_test)
		aux_y = np.concatenate((y_test, sampled_labels), axis=0)

		# see if the discriminator can figure itself out...
		discriminator_test_loss = self.discriminator.evaluate(x, [y, aux_y], verbose=False)

		# make new noise
		noise = np.random.uniform(-1, 1, (2 * num_test, self.latent_size))
		sampled_labels = np.array([self.tags_to_embeddings(self.random_tags()) for _ in range(2 * num_test)])

		trick = np.ones(2 * num_test)

		generator_test_loss = self.combined.evaluate([noise, sampled_labels], [trick, sampled_labels], verbose=False)

		return generator_test_loss, discriminator_test_loss

	def generate(self, count=1, latent_space=None, tags=None) -> list:
		"""
		count: An int that determines the number of examples to generate. Must be 1 if latent_space is set.

		latent_space: A list the size of self.latent_size that contains values from -1 to 1.

		tags: A list of strings that are in self.tags. Size must be less than self.num_classes.

		Returns a list of PIL images.
		"""
		assert isinstance(count, int)
		assert (latent_space != None and count == 1) or (latent_space == None and count >= 1)
		assert latent_space == None or len(latent_space) == self.latent_size
		assert len(tags) <= self.num_classes

		if latent_space == None:
			latent_space = np.random.uniform(-1, 1, (count, self.latent_size))
		else:
			latent_space = [latent_space]

		if tags == None:
			tags = self.random_tags()
			print("using random tags:", tags)
		else:
			print("using provided tags:", tags)
		sampled_labels = np.array([self.tags_to_embeddings(tags) for _ in range(count)])

		generated_images = self.generator.predict([latent_space, sampled_labels], verbose=0)
		generated_images = generated_images * 127.5 + 127.5

		return list([array_to_img(array) for array in generated_images])

class SuperSampler(object):
	def __init__(self):
		self.model = self.build()
		self._pair_generator = self._sample_pair_generator(Path(args.data_dir))
		self.input_size = (32, 32)
		self.output_size = (64, 64)
	
	def build(self):
		supersampler_input_shape = (32, 32, 3)
		supersampler_output_shape = (64, 64, 3)
		super_out_dim = reduce(lambda x, y: x * y, supersampler_output_shape, 1)

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

		_input = Input(shape=supersampler_input_shape, name='input')

		Feature = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
		Feature_out = Res_block()(Feature)

		# Upsampling
		Upsampling1 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
		Upsampling2 = Conv2DTranspose(filters=4, kernel_size=(14, 14), strides=(2, 2), padding='same', activation='relu')(Upsampling1)
		Upsampling3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

		# Mulyi-scale Reconstruction
		Reslayer1 = Res_block()(Upsampling3)

		Reslayer2 = Res_block()(Reslayer1)

		Multi_scale1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Reslayer2)

		Multi_scale2a = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale1)

		Multi_scale2b = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(Multi_scale1)
		Multi_scale2b = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2b)

		Multi_scale2c = Conv2D(filters=16, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu')(Multi_scale1)
		Multi_scale2c = Conv2D(filters=16, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2c)

		Multi_scale2d = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(Multi_scale1)
		Multi_scale2d = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2d)

		Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

		out = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)
		supersampler = Model(inputs=_input, outputs=out)
		supersampler.compile(optimizer="rmsprop", loss='mean_squared_error')
		return supersampler

	def load_checkpoint(self, ckpt_path: Path):
		self.model.load_weights(str(ckpt_path))

	def _sample_pair_generator(self, source_path: Path):
		"""
		Iterates through images from source_path, and returns

		Returns a tuple of 2 3-dimentional arrays representing images.
		"""
		if isinstance(source_path, str):
			source_path = Path(source_path)
		assert isinstance(source_path, Path)

		while True:
			all_image_files = list(source_path.iterdir)
			for img_path in all_image_files:
				img_orig = load_img(img_path)
				img_small = img_orig.copy().resize(self.input_size)

				for y in range(0, img_orig.size[1] - self.output_size[1], self.output_size[1]):
					for x in range(0, img_orig.size[0] - self.output_size[0], self.output_size[0]):
						img_x = img_small.crop((x / 2, y / 2, x + self.input_size[0], y + self.input_size[1]))
						img_y = img_orig.crop((x, y, x + self.output_size[0], y + self.output_size[1]))

						array_x = img_to_array(img_x) / 255
						array_y = img_to_array(img_y) / 255

						yield array_x, array_y

	def get_batch(self, batch_size=2048):
		"""
		batch_size: The number of samples in the batch.

		Returns a batch to train on in the form of a tuple of 2 lists: (X, Y)
		"""
		return zip(*[next(self._pair_generator) for _ in range(batch_size)])

	def train(self, epochs, resume=1, steps_per_epoch=args.steps_per_epoch):
		assert epochs > 0

		for epoch in range(resume, epochs + 1):
			print("Epoch", epoch)
			for step in range(1, steps_per_epoch + 1):
				print("> {}/{} Grabbing batch...".format(step, steps_per_epoch), end='\r')
				data_x, data_y = self.get_batch()
				print("> {}/{} Training...".format(step, steps_per_epoch), end='\r')
				self.model.train_on_batch(data_x, data_y)
			print()

	def upscale(self, orig_img: Image) -> Image:
		"""
		Returns an image that is 2x the resolution of the input image.

		NOTE: There are probably some optimizations that could be made here.
		TODO: agregate all chunks into a list before upscaling them.
		"""
		figure = np.zeros((orig_img.size[1] * 2, orig_img.size[0] * 2, 3))
		new_size = (orig_img.size[0] * 2, orig_img.size[1] * 2)
		
		blend_weights_x = []
		blend_weights_y = []
		for offset in [0, 32]: # this makes the blending take place after the initial prediction
			for y in range(0 + offset, new_size[1], 64 - offset):
				for x in range(0 + offset, new_size[0], 64 - offset):
					img_x = orig_img.crop((int(x/2), int(y/2), int(x/2) + 32, int(y/2) + 32))
					array_x = img_to_array(img_x)
					max_pixel_value = array_x.max()
					array_x /= max_pixel_value

					print("predicting at ", x, ",", y)
					chunks_out = self.model.predict(np.array([array_x]))
					img_y = array_to_img(chunks_out[0])
					if x + 64 > new_size[0]:
						img_y = img_y.crop((0, 0, new_size[0]-x, img_y.size[0]))
					if y + 64 > new_size[1]:
						img_y = img_y.crop((0, 0, img_y.size[1], new_size[1]-y))
					try:
						array_y = img_to_array(img_y)
						array_y *= max_pixel_value

						# image math: blending - https://homepages.inf.ed.ac.uk/rbf/HIPR2/blend.htm
						# https://stackoverflow.com/questions/5919663/how-does-photoshop-blend-two-images-together
						if len(blend_weights_x) == 0 or blend_weights_x.shape != array_y.shape:
							print("recalculate blend_weights_x")
							blend_weights_x = np.array([np.concatenate([np.linspace(0, 1, int(array_y.shape[1] / 2)), np.linspace(1, 0, int(array_y.shape[1] / 2))]) for _ in range(array_y.shape[0])])
							blend_weights_x = blend_weights_x.reshape((*array_y.shape[:-1], 1))
							blend_weights_x = np.repeat(blend_weights_x, repeats=3, axis=2)
						if len(blend_weights_y) == 0 or blend_weights_y.shape != array_y.shape:
							print("recalculate blend_weights_y")
							blend_weights_y = np.array([np.concatenate([np.linspace(0, 1, int(array_y.shape[0] / 2)), np.linspace(1, 0, int(array_y.shape[0] / 2))]) for _ in range(array_y.shape[1])])
							blend_weights_y = np.rot90(blend_weights_y)
							blend_weights_y = blend_weights_y.reshape((*array_y.shape[:-1], 1))
							blend_weights_y = np.repeat(blend_weights_y, repeats=3, axis=2)
							
						if (offset == 32 and x % 32 == 0 and x % 64 != 0) or (offset == 32 and y % 32 == 0 and y % 64 != 0):
							blend_weights = None
							if (x % 32 == 0 and x % 64 != 0) and (y % 32 == 0 and y % 64 != 0):
								blend_weights = blend_weights_x * blend_weights_y
								blend_weights = np.clip(blend_weights, 0, 1)
							elif x % 32 == 0 and x % 64 != 0:
								blend_weights = blend_weights_x
							elif y % 32 == 0 and y % 64 != 0:
								blend_weights = blend_weights_y
							source = figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] * (1 - blend_weights)
							target = array_y * blend_weights
							result = source + target
							figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] = result
						if offset == 0:
							figure[y : y+array_y.shape[0], x : x+array_y.shape[1]] = array_y
							
					except ValueError as e:
						print(x, y, figure.shape, img_y.size, "ValueError, ignoring")
				
		return array_to_img(figure)

def get_latest_epoch(acgan: ACGAN):
	search_for = "params_generator_epoch_*"
	if acgan.latent_size != 100:
		search_for = "params_generator_latent_{}_epoch_*".format(acgan.latent_size)
	matches = sorted(ckpt_dir.glob(search_for))
	print("matches", matches)
	latest_epoch = re.findall(r'[0-9]+', str(matches[-1])[:-5])
	return int(latest_epoch[-1])

if __name__ == "__main__":
	acgan = ACGAN(tags=tags, latent_size=6000)
	supersampler = SuperSampler()

	if args.train:
		if args.train == "acgan":
			acgan.train(args.epochs, resume=args.resume)
		elif args.train == "supersampler":
			if args.resume > 1:
				print("loading supersampler weights")
				supersampler.load_checkpoint(ckpt_dir / "superres_weights.h5")
			supersampler.train(args.epochs, resume=args.resume)

	if args.generate > 0:
		if not args.train:
			target_epoch = get_latest_epoch(acgan)
			print("loading acgan weights from epoch {}".format(target_epoch))
			acgan.load_checkpoint(target_epoch)
			print("loading supersampler weights")
			supersampler.load_checkpoint(ckpt_dir / "superres_weights.h5")
		gen_num = 0
		now = datetime.now()
		for img in acgan.generate(args.generate, tags=args.tags):
			img_up = img
			for _ in range(args.gen_upscale):
				img_up = supersampler.upscale(img_up)
			img_up.save(str(gen_dir / "gen_{}_{}{}{}_{}{}_{}.png".format(acgan._get_file_name_suffix(target_epoch), now.year, now.month, now.day, now.hour, now.minute, gen_num)))
			gen_num += 1
	
	if args.upscale:
		if not (args.generate or args.train):
			print("loading supersampler weights")
			supersampler.load_checkpoint(ckpt_dir / "superres_weights.h5")
		
		img_path = Path(args.upscale)
		img = load_img(img_path)
		target_path = Path("{}_up2{}".format(img_path.stem, img_path.suffix))
		supersampler.upscale(img).save(target_path)

	if args.evaluate:
		target_path = Path(args.evaluate)

		loss_list = []
		for epoch in range(1, get_latest_epoch(acgan) + 1):
			print("Evaluating epoch {}...".format(epoch))
			acgan.load_checkpoint(epoch)
			loss_list.append((epoch, acgan.evaluate()))
		
		print("Saving to {}".format(str(target_path)))
		with target_path.open(mode="w") as f:
			f.write("epcoh,generator,discriminator\n")
			for losses in loss_list:
				f.write("{},{},{}\n".format(*losses))

	if args.visualize == "epochs":
		visualization_path /= "preview/ACGAN"

		noise = np.random.uniform(-1, 1, (1, latent_size))
		if len(args.tags) == 0:
			print("using random tags")
			args.tags = acgan.random_tags()
		else:
			print("using provided tags")
		print("tags: {}".format(args.tags))

		for epoch in range(args.resume, args.epochs + 1):
			print("loading epoch {}".format(epoch))
			acgan.load_checkpoint(epoch)

			print("visualizing epoch {}".format(epoch))
			img = acgan.generate(latent_space=noise, tags=args.tags)[0]
			image_path = visualization_dir / "vis_epoch_{0:04d}.png".format(epoch)
			img.save(str(image_path))

	elif args.visualize == "layers":
		visualization_path /= "layers/ACGAN"

		# visualize each class
		for t in range(len(tags)):
			save_path = visualization_path / "out_{}.png".format(tags[t])
			if save_path.exists():
				# print("{} already visualized".format(tags[t]))
				continue

			output_layer_name = acgan.generator.outputs[0].name
			layer_idx = utils.find_layer_idx(acgan.generator, output_layer_name)

			# Swap softmax with linear
			acgan.generator.layers[layer_idx].activation = activations.linear
			acgan.generator = utils.apply_modifications(acgan.generator)

			img = visualize_activation(acgan.generator, layer_idx, filter_indices=t, verbose=True)
			array_to_img(img).save(save_path)
			print("saved to {}".format(str(save_path)))

		# visualize each layer
		for layer_name in [layer.name for layer in acgan.generator.layers]:
			save_path = visualization_path / "{}.png".format(layer_name)
			if save_path.exists():
				# print("{} already visualized".format(layer_name))
				continue
			if any([x in layer_name for x in ["batch_normalization", "input"]]):
				print("skipping visualization of {}".format(layer_name))
				continue

			# Utility to search for layer index by name. 
			# Alternatively we can specify this as -1 since it corresponds to the last layer.
			layer_idx = utils.find_layer_idx(acgan.generator, layer_name)

			# Swap softmax with linear
			acgan.generator.layers[layer_idx].activation = activations.linear
			acgan.generator = utils.apply_modifications(acgan.generator)

			img = visualize_activation(acgan.generator, layer_idx, verbose=True)
			array_to_img(img).save(save_path)
			print("saved to {}".format(str(save_path)))

	elif args.visualize == "model":
		from keras.models import model_from_json
		from keras.utils import plot_model
		if not (visualization_dir / "ACGAN").exists():
			(visualization_dir / "ACGAN").mkdir(parents=True)
		plot_model(acgan.combined, to_file=str(visualization_dir / "ACGAN" / "combined.png"), show_shapes=True)
		def _visualize_sub_models(model):
			assert isinstance(model, Model) or isinstance(model, Sequential)
			for layer in model.layers:
				if isinstance(layer, Model) or isinstance(layer, Sequential):
					print(layer.name)
					plot_model(layer, to_file=str(visualization_dir / "ACGAN" / "{}.png".format(layer.name)), show_shapes=True)
					_visualize_sub_models(layer)
		_visualize_sub_models(acgan.combined)

	if args.export_model_json:
		assert len(args.export_model_json) > 0
		print("exporting generator as JSON to", args.export_model_json)
		model_json = acgan.generator.to_json()
		with open(args.export_model_json, mode="w") as f:
			f.write(model_json)