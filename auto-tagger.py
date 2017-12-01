#!/usr/bin/env python3
import argparse, json, pickle, random
import numpy as np
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import *
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

tags = ["car","minimal","nature","animal","landscape","people","abstract","city","watermark","text","space","sci-fi","interior","fantasy"]
# tags = ["car","minimal","nature","animal","landscape","people","abstract","city","space","sci-fi","interior","fantasy"]

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--epochs", type=int, default=30)
# parser.add_argument("--resume", type=int, default=1, help="The epoch at which to resume training.")
parser.add_argument("--steps-per-epoch", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)

parser.add_argument("--visualize", action="store_true")

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--data-dir", type=str, default="data/good/img")
parser.add_argument("--tags-file", type=str, default="data/good/tags.csv")
parser.add_argument("--tags-file-out", type=str, default="data/good/tags-auto.csv")

parser.add_argument("img_paths", type=str, nargs="+", default=["auto"])
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

img_width, img_height = 640, 360
in_tags_file = Path(args.tags_file)
out_tags_file = Path(args.tags_file_out)
good_image_path = Path(args.data_dir)
# classes = json.load(open("imagenet1000_clsid_to_human.txt", "r"))
# classes = pickle.load(open("imagenet1000_clsid_to_human.pkl", "rb"))

def getTagsFromFile(img_path):
	with open(str(in_tags_file), "r") as f:
		line = f.readline()
		while line:
			split = line.rstrip("\n").split(",")
			if split[0] == img_path.name:
				return [x for x in split[1:] if x in tags]
			line = f.readline()
	return []

def commitTagsToFile(img_path, img_tags):
	print("commiting tags to file")
	with open(str(self.tags_file), "r") as f:
		lines = f.readlines()
	with open(str(self.tags_file), "w+") as f:
		found = False
		for line in lines:
			split = line.split(",")
			if split[0] == img_path.name:
				print("overwriting line")
				found = True
				f.write(",".join([img_path.name] + img_tags) + "\n")
			else:
				f.write(line)
		if not found:
			print("adding new line")
			f.write(",".join([img_path.name] + img_tags) + "\n")

def get_image_and_tags(img_file):
	return load_img(img_file), getTagsFromFile(img_file)

def tags_to_embeddings(img_tags):
	embeddings = []
	for tag in tags:
		embeddings.append(1 if tag in img_tags else 0)
	return embeddings

# def embeddings_to_tags(embeddings):
# 	tags = []
# 	for i in range(len(tags)):
# 		tags.append()

def preprocess_sample(img, img_tags):
	return img_to_array(img.resize((img_width, img_height))), tags_to_embeddings(img_tags)

def batch_generator():
	data_dir = good_image_path
	all_paths = np.asarray(list(data_dir.iterdir()))
	np.random.shuffle(all_paths)
	batch_start = 0
	# generate batches forever
	while True:
		batch = []
		for i in all_paths[batch_start : batch_start + args.batch_size]:
			img, img_tags = get_image_and_tags(i)
			if len(img_tags) == 0:
				continue
			sample_img, sample_tags = preprocess_sample(img, img_tags)
			yield ({ "input_1": np.asarray([sample_img]) }, { "dense_2": np.asarray([sample_tags]) })
			# batch.append(sample)

		batch_start += len(batch)
		if batch_start >= len(all_paths):
			batch_start = 0

		# yield np.asarray(batch)
		# yield batch
		# yield [ for sample in batch]

def build_model():
	base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_height, img_width, 3), pooling=None, classes=1000)

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have len(tags) classes
	predictions = Dense(len(tags), activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
		layer.trainable = False

	return model

def train_model(model):
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	print("Input", model.input)
	print("Output", model.output)
	earlystop = EarlyStopping(monitor='loss', min_delta=0.1, patience=4, verbose=1, mode='auto')
	model.fit_generator(batch_generator(), steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[earlystop])

	# at this point, the top layers are well trained and we can start fine-tuning
	# convolutional layers from inception V3. We will freeze the bottom N layers
	# and train the remaining top layers.

	# let's visualize layer names and layer indices to see how many layers
	# we should freeze:
	# for i, layer in enumerate(base_model.layers):
	# 	print(i, layer.name)

	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 249 layers and unfreeze the rest:
	for layer in model.layers[:249]:
		layer.trainable = False
	for layer in model.layers[249:]:
		layer.trainable = True

	# we need to recompile the model for these modifications to take effect
	# we use SGD with a low learning rate
	from keras.optimizers import SGD
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

	# we train our model again (this time fine-tuning the top 2 inception blocks
	# alongside the top Dense layers
	earlystop = EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='auto')
	model.fit_generator(batch_generator(), steps_per_epoch=10, epochs=args.epochs, callbacks=[earlystop])

	return model

def visualize(model):
	from vis.visualization import visualize_activation
	from vis.utils import utils
	from keras import activations

	for layer_name in ['dense_2', 'conv2d_20', 'conv2d_22', 'conv2d_25', 'conv2d_26', 'conv2d_85', 'mixed8']:
		# Utility to search for layer index by name. 
		# Alternatively we can specify this as -1 since it corresponds to the last layer.
		layer_idx = utils.find_layer_idx(model, layer_name)

		# Swap softmax with linear
		model.layers[layer_idx].activation = activations.linear
		model = utils.apply_modifications(model)

		# This is the output node we want to maximize.
		filter_idx = 0
		img = visualize_activation(model, layer_idx, filter_indices=filter_idx, verbose=True)
		array_to_img(img).save("visualization/{}.png".format(layer_name))

def predict_for(model, img_path):
	print("predicting for:", img_path)
	test_img = load_img(img_path).resize((img_width, img_height))
	test_img = img_to_array(test_img)
	print("Input shape:", test_img.shape)
	test_img = preprocess_input(test_img)
	print("Predicting...")
	pred_vals = model.predict({"input_1": np.asarray([test_img])})
	# prediction = decode_predictions(pred_vals)
	for pred in pred_vals:
		for i in range(len(tags)):
			print("{} => {:.4%}".format(tags[i], float(pred[i])))
		prediction = list(zip(tags, pred))
		print(prediction)
		return prediction

path_gen = good_image_path.iterdir()
img_paths = args.img_paths
if len(img_paths) == 0 or img_paths[0] == "auto":
	all_paths = list(path_gen)
	img_paths = [random.choice(all_paths) for _ in range(3)]
model = build_model()
model.summary()
if args.train:
	model = train_model(model)
	model.save_weights("ckpt/auto_tagger.h5")
else:
	model.load_weights("ckpt/auto_tagger.h5")

if args.visualize:
	visualize(model)

for img_path_str in img_paths:
	img_path = Path(img_path_str)
	predict_for(model, img_path)