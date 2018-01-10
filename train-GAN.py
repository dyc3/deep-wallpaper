#!/usr/bin/env python3

# found this paper: https://web.stanford.edu/class/cs221/2017/restricted/p-final/dkimball/final.pdf
# code from paper: https://github.com/karenyang/MultiGAN

import h5py, argparse, sys, os, pandas
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from tqdm import tqdm
from collections import defaultdict
import glob, re
try:
	import cPickle as pickle
except ImportError:
	import pickle
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Lambda, Layer, Dropout
from keras.layers import Flatten, Activation, Embedding
from keras.layers import concatenate, add, multiply
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras import metrics
from keras.utils.generic_utils import Progbar
from keras.callbacks import CallbackList, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

tags = ["car","minimal","nature","animal","landscape","people","abstract","city","watermark","text","space","sci-fi","interior","fantasy"]

parser = argparse.ArgumentParser()
parser.add_argument("--clean", help="remove the currently trained model", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--resume", type=int, default=1, help="The epoch at which to resume training. (Will load from the previous epoch)")
parser.add_argument("--steps-per-epoch", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=32)

parser.add_argument("--data-dir", type=str, default="data/good/img")
parser.add_argument("--tags-file", type=str, default="data/good/tags.csv")

parser.add_argument("--export-full-model", type=str)
parser.add_argument("--export-model-json", type=str)

parser.add_argument("--generate", type=int, default=0)
parser.add_argument("--tags", nargs="+", help="Specify tags to use when generating images, otherwise random tags will be used.", default=[])

parser.add_argument("--visualize", type=str, choices=["epochs", "layers"])
args = parser.parse_args()

img_width, img_height, img_chns = 128, 72, 3
tags_file = Path("data/good/tags.csv")

visualization_dir = Path("visualization/preview/ACGAN/")
if not visualization_dir.exists():
	visualization_dir.mkdir(parents=True)

ckpt_dir = Path("ckpt/")
if not ckpt_dir.exists():
	ckpt_dir.mkdir(parents=True)

gen_dir = Path("gen/")
if not gen_dir.exists():
	gen_dir.mkdir(parents=True)

def load_tags():
	tags_file = Path(args.tags_file)
	tags = pandas.read_csv(tags_file, names=["file", "tagA", "tagB", "tagC", "tagD", "tagE"])
	return tags

def getTagsFromFile(img_path):
	with open(str(tags_file), "r") as f:
		line = f.readline()
		while line:
			split = line.rstrip("\n").split(",")
			if split[0] == img_path.name:
				return [x for x in split[1:] if x in tags]
			line = f.readline()
	return []

def get_image_and_tags(img_file):
	assert isinstance(img_file, str) or isinstance(img_file, Path)
	if isinstance(img_file, str):
		img_file = Path(img_file)
	assert img_file.exists(), "{} does not exist.".format(img_file)
	return load_img(img_file), getTagsFromFile(img_file)

def tags_to_index(in_tags):
	indices = []
	for tag in in_tags:
		if tag in tags:
			indices.append(tags.index(tag))
	return np.array(indices)

def tags_to_embeddings(img_tags):
	embeddings = []
	for tag in tags:
		embeddings.append(1 if tag in img_tags else 0)
	return np.array(embeddings, dtype="float32")

def random_tags():
	return np.random.choice(tags, size=np.random.randint(1, len(tags)))

def get_latest_epoch():
	matches = sorted(ckpt_dir.glob("params_generator_epoch_*"))
	latest_epoch = re.findall(r'[0-9]+', str(matches[-1]))
	return int(latest_epoch[0])

def batch_generator():
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
			img_tags = getTagsFromFile(img_path)
			checked += 1
			if len(img_tags) == 0:
				continue
			img_tags = tags_to_embeddings(img_tags)
			img = img_to_array(load_img(img_path).resize((img_width, img_height)))
			img /= 127.5
			img -= 1
			batch.append((img, img_tags))

		batch_start += checked
		if batch_start >= len(all_paths):
			batch_start = 0

		yield np.asarray(batch)

def build_generator(latent_size=100, num_classes=2):
	assert latent_size > 0
	assert num_classes > 0
	# we will map a pair of (z, L), where z is a latent vector and L is a
	# label drawn from P_c, to image space (..., 28, 28, 1)
	cnn = Sequential()

	cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
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
	
	cnn.summary()

	# this is the z space commonly refered to in GAN papers
	latent = Input(shape=(latent_size, ), dtype="float32")

	# this will be our label
	image_class = Input(shape=(num_classes,), dtype="float32")

	cls = Flatten()(Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class))
	cls = Dense(latent_size)(cls)

	# hadamard product between z-space and a class conditional embedding
	h = multiply([latent, cls])

	fake_image = cnn(h)

	return Model([latent, image_class], fake_image)

def build_discriminator(num_classes=2):
	assert num_classes > 0
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
	aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

	return Model(image, [fake, aux])

def build_combined(generator, discriminator, latent_size=100, num_classes=2):
	assert latent_size > 0
	assert num_classes > 0

	# Adam parameters suggested in https://arxiv.org/abs/1511.06434
	adam_lr = 0.0002
	adam_beta_1 = 0.5

	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'categorical_crossentropy'])

	latent = Input(shape=(latent_size, ))
	image_class = Input(shape=(num_classes,), dtype='float32')

	# get a fake image
	fake = generator([latent, image_class])

	# we only want to be able to train generation for the combined model
	discriminator.trainable = False
	fake, aux = discriminator(fake)
	combined = Model([latent, image_class], [fake, aux])

	print('Combined model:')
	combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'categorical_crossentropy'])
	combined.summary()

	return combined

def train(generator, discriminator, latent_size=100, num_classes=2):
	batch_gen = batch_generator()
	train_history = defaultdict(list)
	test_history = defaultdict(list)

	for epoch in range(args.resume, args.epochs + 1):
		progress_bar = Progbar(target=args.steps_per_epoch * args.batch_size)

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
			noise = np.random.uniform(-1, 1, (args.batch_size, latent_size))

			# sample some labels from p_c
			sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(args.batch_size)])

			# generate a batch of fake images, using the generated labels as a
			# conditioner. We reshape the sampled labels to be
			# (batch_size, 1) so that we can feed them into the embedding
			# layer as a length one sequence
			generated_images = generator.predict([noise, sampled_labels], verbose=0)

			x = np.concatenate((image_batch, generated_images))

			# use soft real/fake labels
			soft_zero, soft_one = 0.25, 0.75
			y = np.array([soft_one] * this_batch_size + [soft_zero] * args.batch_size)
			y = y.reshape(-1, 1)
			aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

			# see if the discriminator can figure itself out...
			epoch_disc_loss.append(discriminator.train_on_batch(x, {"generation":y, "auxiliary":aux_y}))

			# make new noise. we generate args.batch_size + this_batch_size here such that we have
			# the generator optimize over an identical number of images as the
			# discriminator
			noise = np.random.uniform(-1, 1, (args.batch_size + this_batch_size, latent_size))
			sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(args.batch_size + this_batch_size)])

			# we want to train the generator to trick the discriminator
			# For the generator, we want all the {fake, not-fake} labels to say
			# not-fake
			trick = np.ones(args.batch_size + this_batch_size) * soft_one

			epoch_gen_loss.append(combined.train_on_batch(
				[noise, sampled_labels],
				[trick, sampled_labels]))

			progress_bar.update(index * args.batch_size)

		print('Testing for epoch {}:'.format(epoch))

		# evaluate the testing loss here

		# generate a new batch of noise
		num_test = args.batch_size
		x_test, y_test = zip(*next(batch_gen))
		noise = np.random.uniform(-1, 1, (num_test, latent_size))

		# sample some labels from p_c and generate images from them
		sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(args.batch_size)])
		generated_images = generator.predict([noise, sampled_labels], verbose=False)

		x = np.concatenate((x_test, generated_images))
		y = np.array([1] * len(x_test) + [0] * num_test)
		aux_y = np.concatenate((y_test, sampled_labels), axis=0)

		# see if the discriminator can figure itself out...
		discriminator_test_loss = discriminator.evaluate(x, [y, aux_y], verbose=False)

		discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

		# make new noise
		noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
		sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(2 * num_test)])

		trick = np.ones(2 * num_test)

		generator_test_loss = combined.evaluate(
			[noise, sampled_labels],
			[trick, sampled_labels], verbose=False)

		generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

		# generate an epoch report on performance
		train_history['generator'].append(generator_train_loss)
		train_history['discriminator'].append(discriminator_train_loss)

		test_history['generator'].append(generator_test_loss)
		test_history['discriminator'].append(discriminator_test_loss)

		print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
			'component', *discriminator.metrics_names))
		print('-' * 65)

		ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
		print(ROW_FMT.format('generator (train)',
							 *train_history['generator'][-1]))
		print(ROW_FMT.format('generator (test)',
							 *test_history['generator'][-1]))
		print(ROW_FMT.format('discriminator (train)',
							 *train_history['discriminator'][-1]))
		print(ROW_FMT.format('discriminator (test)',
							 *test_history['discriminator'][-1]))

		# save weights every epoch
		generator.save_weights(str(ckpt_dir / 'params_generator_epoch_{0:04d}.hdf5'.format(epoch)), True)
		discriminator.save_weights(str(ckpt_dir / 'params_discriminator_epoch_{0:04d}.hdf5'.format(epoch)), True)

		# generate some digits to display
		num_rows = 10
		noise = np.random.uniform(-1, 1, (num_rows * num_classes, latent_size))

		sampled_labels = np.array([
			list(tags_to_embeddings([tag])) * num_rows for tag in tags
		]).reshape(-1, num_classes)

		# get a batch to display
		generated_images = generator.predict([noise, sampled_labels], verbose=0)

		# prepare real images not sorted by class label
		# real_images, real_labels = zip(*next(batch_gen))
		real_images, real_labels = x_test, y_test
		indices = np.argsort(real_labels, axis=0)

		# display generated images, white separator, real images
		img = np.concatenate(
			(generated_images,
			 real_images))

		# arrange them into a grid
		print(img.shape)
		img = np.concatenate(np.hstack(np.split(img, 43)), axis=1) * 127.5 + 127.5
		preview_img_path = str(visualization_dir / 'plot_epoch_{0:04d}_generated.png'.format(epoch))
		array_to_img(img).save(preview_img_path)
		print("saved preview to {}".format(preview_img_path))

	pickle.dump({'train': train_history, 'test': test_history}, open('acgan-history.pkl', 'wb'))

print("args: {}".format(args))

generator, discriminator = build_generator(num_classes=len(tags)), build_discriminator(num_classes=len(tags))
print("Generator:")
generator.summary()
print("Discriminator:")
discriminator.summary()
combined = build_combined(generator, discriminator, num_classes=len(tags))

if args.export_full_model:
	assert len(args.export_full_model) > 0
	target_epoch = get_latest_epoch()
	print("exporting model from epoch {}".format(target_epoch))
	generator.load_weights(str(ckpt_dir / 'params_generator_epoch_{0:04d}.hdf5'.format(target_epoch)), True)
	generator.save(args.export_full_model.format(target_epoch))

if args.export_model_json:
	assert len(args.export_model_json) > 0
	print("exporting model as JSON")
	model_json = generator.to_json()
	with open(args.export_model_json, mode="w") as f:
		f.write(model_json)

if args.train:
	np.random.seed(42)
	if args.resume > 1:
		generator.load_weights(str(ckpt_dir / 'params_generator_epoch_{0:04d}.hdf5'.format(args.resume-1)), True)
		discriminator.load_weights(str(ckpt_dir / 'params_discriminator_epoch_{0:04d}.hdf5'.format(args.resume-1)), True)
	train(generator, discriminator, num_classes=len(tags))

if args.generate:
	latent_size = 100
	target_epoch = get_latest_epoch()
	print("loading generator from epoch {}".format(target_epoch))
	generator.load_weights(str(ckpt_dir / 'params_generator_epoch_{0:04d}.hdf5'.format(target_epoch)), True)

	noise = np.random.uniform(-1, 1, (args.generate, latent_size))
	if len(args.tags) == 0:
		print("using random tags")
		sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(args.generate)])
	else:
		print("using provided tags:", args.tags)
		sampled_labels = np.array([tags_to_embeddings(args.tags) for _ in range(args.generate)])
	generated_images = generator.predict([noise, sampled_labels], verbose=0)
	generated_images = generated_images * 127.5 + 127.5
	
	for array in generated_images:
		i = 0
		while (gen_dir / "gen{}.png".format(i)).exists():
			i += 1
		array_to_img(array).save(str(gen_dir / "gen{}.png".format(i)))

if args.visualize == "epochs":
	latent_size = 100
	noise = np.random.uniform(-1, 1, (1, latent_size))
	if len(args.tags) == 0:
		print("using random tags")
		sampled_labels = np.array([tags_to_embeddings(random_tags()) for _ in range(1)])
	else:
		print("using provided tags:", args.tags)
		sampled_labels = np.array([tags_to_embeddings(args.tags) for _ in range(1)])
	for epoch in range(args.epochs):
		print("visualizing epoch {}".format(epoch))
		current_model_file = ckpt_dir / 'params_generator_epoch_{0:04d}.hdf5'.format(epoch)
		if not current_model_file.exists():
			print("missing checkpoint for epoch {}".format(epoch))
			continue

		generator.load_weights(str(current_model_file), True)
		generated_images = generator.predict([noise, sampled_labels], verbose=0)
		generated_images = generated_images * 127.5 + 127.5
		
		for array in generated_images:
			image_path = visualization_dir / "vis_epoch_{0:04d}.png".format(epoch)
			array_to_img(array).save(str(image_path))
elif args.visualize == "layers":
	print("Not yet implemented")
