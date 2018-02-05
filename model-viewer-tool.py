#!/usr/bin/env python3
import sys, shutil, os, random
from pathlib import Path
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Ui_ModelViewer import Ui_MainWindow
from keras.models import load_model, model_from_json
from keras.preprocessing.image import array_to_img, img_to_array

class SliderFeature(QSlider):
	def __init__(self, parent):
		super(SliderFeature, self).__init__(parent)

	def mousePressEvent(self, event):
		if event.button() == Qt.RightButton:
			self.setValue(0);
		super().mousePressEvent(event)

class ModelViewerWindow(Ui_MainWindow):
	def __init__(self):
		super(ModelViewerWindow, self).__init__()

		self.imgQ = None
		self.model = None
		self.input_sliders = []		
		
	def setup(self):
		self.actionQuit.triggered.connect(self.action_quit_triggered)
		self.actionRandomize.triggered.connect(self.action_randomize_triggered)
		self.actionReset.triggered.connect(self.action_reset_triggered)

		# self.graphicsView
		self.scene = QGraphicsScene()
		rect = self.graphicsView.rect()
		print(rect)
		print(self.graphicsView.frameRect())
		self.scene.setSceneRect(QRectF(rect))
		self.graphicsView.setScene(self.scene)

	def action_quit_triggered(self):
		sys.exit(0)

	def displayImage(self, img):
		self.scene.clear()
		w, h = img.size
		self.imgQ = ImageQt(img) # we need to hold reference to imgQ, or it will crash
		pixMap = QPixmap.fromImage(self.imgQ)
		self.scene.addPixmap(pixMap)
		self.fixImageSize()
		self.scene.update()

	def fixImageSize(self):
		if self.imgQ:
			rectf = QRectF(self.imgQ.rect())
			self.scene.setSceneRect(rectf)
			self.graphicsView.fitInView(rectf, Qt.KeepAspectRatio)

	def resizeEvent(self, e):
		# print(self.imgQ.rect())
		# rectf = QRectF(self.scene.sceneRect())
		self.fixImageSize()
		super().resizeEvent(e)

	def load_model(self, model_path, model_weights_path=None):
		if model_weights_path:
			print("loading model from json:", model_path)
			with open(model_path) as f:
				self.model = model_from_json("".join(f.readlines()))
			print("loading weights from", model_weights_path)
			self.model.load_weights(str(model_weights_path))
		else:
			print("loading full model:", model_path)
			self.model = load_model(str(model_path))
		self.update_gui_for_model()
		
	def update_gui_for_model(self):
		self.input_sliders = []
		for input_layer in self.model.inputs:
			print(input_layer)
			print(input_layer.get_shape())
			grpInputTemplate = QGroupBox(self.scrInputsContents)
			grpInputTemplate.setObjectName("grpInputTemplate")
			verticalLayout_2 = QVBoxLayout(grpInputTemplate)
			verticalLayout_2.setContentsMargins(0, 0, 0, 0)
			verticalLayout_2.setSpacing(0)
			verticalLayout_2.setObjectName("verticalLayout_2")

			lblInputName = QLabel(grpInputTemplate)
			lblInputName.setText(input_layer.name)
			sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
			sizePolicy.setHorizontalStretch(0)
			sizePolicy.setVerticalStretch(0)
			sizePolicy.setHeightForWidth(lblInputName.sizePolicy().hasHeightForWidth())
			lblInputName.setSizePolicy(sizePolicy)
			lblInputName.setMinimumSize(QSize(20, 0))
			lblInputName.setAlignment(Qt.AlignCenter)
			lblInputName.setObjectName("lblInputName_{}".format(input_layer.name))

			layer_sliders = []
			for feature in range(input_layer.get_shape()[1]):
				frFeatureTemplate = QFrame(grpInputTemplate)
				sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
				sizePolicy.setHorizontalStretch(0)
				sizePolicy.setVerticalStretch(0)
				sizePolicy.setHeightForWidth(frFeatureTemplate.sizePolicy().hasHeightForWidth())
				frFeatureTemplate.setSizePolicy(sizePolicy)
				frFeatureTemplate.setFrameShape(QFrame.StyledPanel)
				frFeatureTemplate.setFrameShadow(QFrame.Raised)
				frFeatureTemplate.setObjectName("frInputFeature_{}_{}".format(input_layer.name, feature))
				horizontalLayout = QHBoxLayout(frFeatureTemplate)
				horizontalLayout.setObjectName("horizontalLayout")

				lblFeatureIndex = QLabel(frFeatureTemplate)
				sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
				sizePolicy.setHorizontalStretch(0)
				sizePolicy.setVerticalStretch(0)
				sizePolicy.setHeightForWidth(lblFeatureIndex.sizePolicy().hasHeightForWidth())
				lblFeatureIndex.setSizePolicy(sizePolicy)
				lblFeatureIndex.setMinimumSize(QSize(20, 0))
				lblFeatureIndex.setAlignment(Qt.AlignCenter)
				lblFeatureIndex.setObjectName("lblFeatureIndex_{}_{}".format(input_layer.name, feature))
				lblFeatureIndex.setText(str(feature))
				horizontalLayout.addWidget(lblFeatureIndex)

				sliderFeatureValue = SliderFeature(frFeatureTemplate)
				sliderFeatureValue.setMinimumSize(QSize(100, 0))
				sliderFeatureValue.setOrientation(Qt.Horizontal)
				sliderFeatureValue.setMinimum(-1000)
				sliderFeatureValue.setMaximum(1000)
				sliderFeatureValue.setObjectName("sliderFeatureValue_{}_{}".format(input_layer.name, feature))
				sliderFeatureValue.valueChanged.connect(self.sliderFeatureValue_onValueChanged)
				sliderFeatureValue.sliderReleased.connect(self.sliderFeatureValue_onSliderReleased)
				horizontalLayout.addWidget(sliderFeatureValue)
				layer_sliders.append(sliderFeatureValue)

				lblFeatureValue = QLabel(frFeatureTemplate)
				sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
				sizePolicy.setHorizontalStretch(0)
				sizePolicy.setVerticalStretch(0)
				sizePolicy.setHeightForWidth(lblFeatureValue.sizePolicy().hasHeightForWidth())
				lblFeatureValue.setSizePolicy(sizePolicy)
				lblFeatureValue.setMinimumSize(QSize(40, 0))
				lblFeatureValue.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
				lblFeatureValue.setObjectName("lblFeatureValue_{}_{}".format(input_layer.name, feature))
				lblFeatureValue.setText("0")
				horizontalLayout.addWidget(lblFeatureValue)

				verticalLayout_2.addWidget(frFeatureTemplate)
			self.vertInputs.addWidget(grpInputTemplate)
			self.input_sliders.append(layer_sliders)

	def sliderFeatureValue_onValueChanged(self, value):
		# print(type(self), self, type(value), value)
		for layer_sliders in self.input_sliders:
			for slider in layer_sliders:
				label = slider.parent().findChildren(QLabel, slider.objectName().replace("slider", "lbl"))[0]
				label.setText(str(slider.value() / 1000))

	def sliderFeatureValue_onSliderReleased(self):
		self.updateOutput()

	def updateOutput(self):
		inputs = [np.asarray([slider.value() / 1000 for slider in layer_sliders]).reshape(-1, len(layer_sliders)) for layer_sliders in self.input_sliders]
		print(inputs)
		self.renderInputs(inputs)

	def renderInputs(self, inputs):
		array = self.model.predict(inputs)[0]
		img = array_to_img(array)
		self.displayImage(img)

	def action_randomize_triggered(self):
		print("randomize inputs")
		for layer_sliders in self.input_sliders:
			for slider in layer_sliders:
				slider.setValue(random.randint(slider.minimum(), slider.maximum()))
		self.updateOutput()

	def action_reset_triggered(self):
		print("reset inputs")
		for layer_sliders in self.input_sliders:
			for slider in layer_sliders:
				slider.setValue(0)
		self.updateOutput()
	

if __name__ == "__main__":
	print("args:", sys.argv)
	app = QApplication(sys.argv)
	MainWindow = QMainWindow()
	ui = ModelViewerWindow()
	ui.setupUi(MainWindow)
	ui.setup()
	
	MainWindow.show()
	if len(sys.argv) == 2:
		ui.load_model(sys.argv[1])
	elif len(sys.argv) == 3:
		ui.load_model(sys.argv[1], sys.argv[2])
	
	sys.exit(app.exec_())