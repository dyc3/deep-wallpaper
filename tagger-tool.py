#!/usr/bin/env python3
import sys, shutil, os
from pathlib import Path
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Ui_DataTaggerWindow import Ui_MainWindow

tags = ["car","minimal","nature","animal","landscape","people","abstract","city","watermark","text","space","sci-fi","interior","fantasy"]

class DataTaggerWindow(Ui_MainWindow):
	"""docstring for DataTaggerWindow."""
	def __init__(self):
		super(DataTaggerWindow, self).__init__()
		# self._mainWindow = QMainWindow()
		self.setupUi()

		self.raw_image_path = Path("data/raw")
		self.good_image_path = Path("data/good/img")
		self.bad_image_path = Path("data/bad/img")
		self.tags_file = Path("data/good/tags.csv")
		self.path_gen = self.good_image_path.iterdir()

		if not self.good_image_path.exists():
			self.good_image_path.mkdir(parents=True)
		if not self.bad_image_path.exists():
			self.bad_image_path.mkdir(parents=True)

		self.last_image_path = None
		self.current_image_path = None

		self.setup()

	def setup(self):
		# self.graphicsView
		self.scene = QGraphicsScene()
		rect = self.graphicsView.rect()
		print(rect)
		print(self.graphicsView.frameRect())
		self.scene.setSceneRect(QRectF(rect))
		self.graphicsView.setScene(self.scene)
		# self.btnGood.clicked.connect(self.onClickGood)
		# self.btnBad.clicked.connect(self.onClickBad)

		# Set up tag buttons
		self.btnTagTemplate.setParent(None) # Hide the template
		font = QFont()
		font.setPointSize(16)
		self.tag_buttons = []
		for tag in tags:
			tag_btn = QPushButton(self.centralwidget)
			tag_btn.setMinimumSize(QSize(0, 20))
			tag_btn.setFont(font)
			tag_btn.setCheckable(True)
			tag_btn.setChecked(False)
			tag_btn.setObjectName("btnTag_{}".format(tag))
			tag_btn.setText(tag)
			self.tag_buttons.append(tag_btn)
			self.hlayTags.addWidget(tag_btn)

		self.nextImg()

	def nextImg(self):
		print("finding next image")
		# current_subdir = None
		# while not current_subdir or len(list(current_subdir.iterdir())) == 0:
		# 	current_subdir = next(self.raw_image_path.iterdir())
		# 	print("current_subdir: {}".format(current_subdir))
		# 	if len(list(current_subdir.iterdir())) == 0:
		# 		print("deleting empty directory")
		# 		os.rmdir(str(current_subdir))
		# 		current_subdir = None
		# 		continue
		self.last_image_path = self.current_image_path
		self.current_image_path = next(self.path_gen)
		print("next image: {}".format(self.current_image_path))
		img = Image.open(self.current_image_path)
		img = img.convert("RGB")
		width, height = img.size
		print("img size: {}x{}".format(*img.size))
		self.displayImage(img)
		img.close()

		# read tags
		img_tags = self.getTagsFromFile(self.current_image_path)
		print("img tags: {}".format(img_tags))
		self.applyTagsToButtons(img_tags)

	def displayImage(self, img):
		self.scene.clear()
		w, h = img.size
		self.imgQ = ImageQt(img) # we need to hold reference to imgQ, or it will crash
		pixMap = QPixmap.fromImage(self.imgQ)
		self.scene.addPixmap(pixMap)
		self.fixImageSize()
		self.scene.update()

	def fixImageSize(self):
		rectf = QRectF(self.imgQ.rect())
		self.scene.setSceneRect(rectf)
		self.graphicsView.fitInView(rectf, Qt.KeepAspectRatio)

	def resizeEvent(self, e):
		# print(self.imgQ.rect())
		# rectf = QRectF(self.scene.sceneRect())
		self.fixImageSize()
		super().resizeEvent(e)

	def keyPressEvent(self, e):
		# key = e.key()
		tag_hotkeys = [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T, Qt.Key_Y, Qt.Key_U, Qt.Key_I, Qt.Key_O, Qt.Key_P, Qt.Key_BracketLeft, Qt.Key_BracketRight]
		if e.key() in tag_hotkeys:
			idx = tag_hotkeys.index(e.key())
			self.tag_buttons[idx].toggle()

		if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
			img_tags = self.getTagsFromButtons()
			self.commitTagsToFile(self.current_image_path, img_tags)
			self.nextImg()
		elif e.key() == Qt.Key_Space:
			print("skipping")
			self.nextImg()
		elif e.key() == Qt.Key_Backspace:
			self.undo()

	def getTagsFromFile(self, img_path):
		with open(str(self.tags_file), "r") as f:
			line = f.readline()
			while line:
				split = line.rstrip("\n").split(",")
				if split[0] == img_path.name:
					return split[1:]
				line = f.readline()
		return []

	def commitTagsToFile(self, img_path, img_tags):
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

	def getTagsFromButtons(self):
		img_tags = []
		for child in self.tag_buttons:
			if child.isChecked():
				img_tags.append(child.text())
		return img_tags

	def applyTagsToButtons(self, img_tags):
		for child in self.tag_buttons:
			child.setChecked(child.text() in img_tags)

	def undo(self):
		if not self.last_image_path:
			print("Can't undo")
			return
		print("going back {} -> {}".format(self.current_image_path, self.last_image_path))
		self.current_image_path = self.last_image_path
		self.nextImg()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	# MainWindow = QMainWindow()
	ui = DataTaggerWindow()
	# ui.setupUi(MainWindow)
	ui.show()
	# ui.setup()
	sys.exit(app.exec_())
