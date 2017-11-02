from pathlib import Path
import json, shutil, os, subprocess

def distance(img_path1, img_path2):
	"""
	A really hacky and lazy way to get the "distance" of images using somebody else's docker container.
	"""
	assert isinstance(img_path1, str) or isinstance(img_path1, Path)
	assert isinstance(img_path2, str) or isinstance(img_path2, Path)

	if isinstance(img_path1, str):
		img_path1 = Path(img_path1)

	if isinstance(img_path2, str):
		img_path2 = Path(img_path2)

	assert img_path1.exists() and img_path2.exists()

	docker_cmd = "docker run --rm -i -v {}/shared:/shared deepaiorg/image-similarity".format(os.getcwd())
	shutil.copy(str(img_path1), "shared/in1")
	shutil.copy(str(img_path2), "shared/in2")

	subprocess.call(docker_cmd.split(" "))

	results = json.load(open("shared/out1", "r"))
	return results["distance"]

def obj_recog(img_path):
	assert isinstance(img_path1, str) or isinstance(img_path1, Path)

	if isinstance(img_path, str):
		img_path = Path(img_path)

	assert img_path1.exists()

	shutil.copy(str(img_path), "shared/in1")

	docker_cmd = "docker run --rm -i -v {}/shared:/shared deepaiorg/densecap".format(os.getcwd())

	results = json.load(open("shared/out1", "r"))
	return results
