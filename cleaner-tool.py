#!/usr/bin/env python3
import sys, shutil, os
from pathlib import Path
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Ui_DataCleanerWindow import Ui_MainWindow

class DataCleanerWindow(Ui_MainWindow):
	"""docstring for DataCleanerWindow."""
	def __init__(self):
		super(DataCleanerWindow, self).__init__()
		# self._mainWindow = QMainWindow()
		self.setupUi()

		self.raw_image_path = Path("data/raw")
		self.good_image_path = Path("data/good/img")
		self.bad_image_path = Path("data/bad/img")

		if not self.good_image_path.exists():
			self.good_image_path.mkdir(parents=True)
		if not self.bad_image_path.exists():
			self.bad_image_path.mkdir(parents=True)

		self.last_image_path = None
		self.last_target_path = None
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
		self.btnGood.clicked.connect(self.onClickGood)
		self.btnBad.clicked.connect(self.onClickBad)

		self.nextImg()

	def nextImg(self):
		print("finding next image")
		current_subdir = None
		while not current_subdir or len(list(current_subdir.iterdir())) == 0:
			current_subdir = next(self.raw_image_path.iterdir())
			print("current_subdir: {}".format(current_subdir))
			if len(list(current_subdir.iterdir())) == 0:
				print("deleting empty directory")
				os.rmdir(str(current_subdir))
				current_subdir = None
				continue
		self.current_image_path = next(current_subdir.iterdir())
		print("next image: {}".format(self.current_image_path))
		img = Image.open(self.current_image_path)
		img = img.convert("RGB")
		width, height = img.size
		print("img size: {}x{}".format(*img.size))
		if height > width or abs(width - height) < 250:
			self.displayImage(img)
			img.close()
			print("Auto trashing image because height > width or image is almost a square")
			self.onClickBad(set_last_image=False)
			self.nextImg()
			return
		else:
			self.displayImage(img)
			img.close()

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
		if e.key() == Qt.Key_Y:
			self.onClickGood()
		elif e.key() == Qt.Key_N:
			self.onClickBad()
		elif e.key() == Qt.Key_Backspace:
			self.undo()

	def onClickGood(self, set_last_image=True):
		if self.current_image_path:
			if (self.good_image_path / self.current_image_path.name).exists():
				print("Already exists, deleting copy")
				os.remove(str(self.current_image_path))
			else:
				if set_last_image:
					self.last_image_path = self.current_image_path
					self.last_target_path = self.good_image_path / self.current_image_path.name
				print("moving {} -> {}".format(self.current_image_path, self.good_image_path))
				shutil.move(str(self.current_image_path), str(self.good_image_path))
		self.nextImg()

	def onClickBad(self, set_last_image=True):
		if self.current_image_path:
			if (self.bad_image_path / self.current_image_path.name).exists():
				print("Already exists, deleting copy")
				os.remove(str(self.current_image_path))
			else:
				if set_last_image:
					self.last_image_path = self.current_image_path
					self.last_target_path = self.bad_image_path / self.current_image_path.name
				print("moving {} -> {}".format(self.current_image_path, self.bad_image_path))
				shutil.move(str(self.current_image_path), str(self.bad_image_path))
		self.nextImg()

	def undo(self):
		if not self.last_image_path:
			print("Can't undo")
			return
		print("Undoing last decision: move file back {} -> {}".format(self.last_target_path, self.last_image_path))
		shutil.move(str(self.last_target_path), str(self.last_image_path))
		self.current_image_path = self.last_image_path
		self.nextImg()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	# MainWindow = QMainWindow()
	ui = DataCleanerWindow()
	# ui.setupUi(MainWindow)
	ui.show()
	# ui.setup()
	sys.exit(app.exec_())
