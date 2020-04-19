# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os, sys, traceback, time 
import numpy as np
import math
import argparse
from PIL import Image
from Worker import *


def img_read(path):
    return Image.open(path).convert('RGB')

def disparity_read(path):
    return Image.open(path)

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(530, 510)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        self.img_height = 180
        self.img_width  = 320

        #
        self.list_image = QtWidgets.QListWidget(self.centralWidget)
        self.list_image.setGeometry(QtCore.QRect(10, 180, 111, 241))
        self.list_image.setObjectName("listView")

        # Buttons
        self.btn_browse = QtWidgets.QPushButton(self.centralWidget)
        self.btn_browse.setGeometry(QtCore.QRect(10, 10, 50, 21))
        self.btn_browse.setObjectName("openButton")

        self.btn_close = QtWidgets.QPushButton(self.centralWidget)
        self.btn_close.setGeometry(QtCore.QRect(10, 430, 50, 21))
        self.btn_close.setObjectName("closeButton")

        self.btn_eval1 = QtWidgets.QPushButton(self.centralWidget)
        self.btn_eval1.setGeometry(QtCore.QRect(10, 100, 50, 21))
        self.btn_eval1.setObjectName("eval1Button")

        self.btn_evalall = QtWidgets.QPushButton(self.centralWidget)
        self.btn_evalall.setGeometry(QtCore.QRect(70, 100, 50, 21))
        self.btn_evalall.setObjectName("evalallButton")

        # labels
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_3.setGeometry(QtCore.QRect(160, 300, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.centralWidget)
        self.label_5.setGeometry(QtCore.QRect(160, 10, 61, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralWidget)
        self.label_6.setGeometry(QtCore.QRect(160, 150, 61, 16))
        self.label_6.setObjectName("label_6")

        # views
        self.v_left = QtWidgets.QLabel(self.centralWidget)
        self.v_left.setGeometry(QtCore.QRect(160, 40, 330, 100))
        self.v_left.setText("")
        self.v_left.setObjectName("v_left")
        self.v_right = QtWidgets.QLabel(self.centralWidget)
        self.v_right.setGeometry(QtCore.QRect(160, 180, 330, 100))
        self.v_right.setText("")
        self.v_right.setObjectName("v_right")
        self.v_disp = QtWidgets.QLabel(self.centralWidget)
        self.v_disp.setGeometry(QtCore.QRect(160, 320, 330, 100))
        self.v_disp.setText("")
        self.v_disp.setObjectName("v_disp")

        # Process indicator
        self.progressBar = QtWidgets.QProgressBar(self.centralWidget)
        self.progressBar.setGeometry(QtCore.QRect(390, 431, 111, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.label_7 = QtWidgets.QLabel(self.centralWidget)
        self.label_7.setGeometry(QtCore.QRect(316, 430, 61, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralWidget)
        self.label_8.setGeometry(QtCore.QRect(10, 150, 61, 20))
        self.label_8.setObjectName("label_8")

        # initialize
        self.v_left.raise_()
        self.v_right.raise_()
        self.v_disp.raise_()
        self.label_3.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.list_image.raise_()
        self.btn_browse.raise_()
        self.btn_close.raise_()
        self.btn_eval1.raise_()
        self.btn_evalall.raise_()
        self.progressBar.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.progressBar.setValue(0)

        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 523, 20))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        self.statusBar.addPermanentWidget(self.label_7)
        self.statusBar.addPermanentWidget(self.progressBar)
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.function_initialization()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Stereo Mathching"))
        self.btn_browse.setText(_translate("MainWindow", "Browse"))
        self.btn_close.setText(_translate("MainWindow", "Close"))
        self.btn_eval1.setText(_translate("MainWindow", "One"))
        self.btn_evalall.setText(_translate("MainWindow", "All"))
        self.label_3.setText(_translate("MainWindow", "Disparity"))
        self.label_5.setText(_translate("MainWindow", "Left Image"))
        self.label_6.setText(_translate("MainWindow", "Right Image"))
        self.label_7.setText(_translate("MainWindow", ""))
        self.label_8.setText(_translate("MainWindow", "Image List"))
    
    def function_initialization(self):
        self.idx = 0
        self.eval_all = False

        # initialize functions for buttons
        self.btn_close.clicked.connect(lambda: sys.exit())
        self.btn_browse.clicked.connect(self.on_browseButton_clicked)
        self.list_image.itemClicked.connect(self.image_item_clicked)
        self.btn_eval1.clicked.connect(lambda: self.evaluate(all=False))
        self.btn_evalall.clicked.connect(lambda: self.evaluate(all=True))

        # create thread pool
        self.threadpool = QThreadPool()

    def image_item_clicked(self, item):
        selected_item_str = str(item.text())
        if selected_item_str in self.left:
            self.idx = self.left.index(selected_item_str)

        # Display the selected image right away
        self.display_seleted_image()

    def on_browseButton_clicked(self):
        
        browse_path = os.path.normpath(QFileDialog.getExistingDirectory(self, 'Select a directory', '', QFileDialog.ShowDirsOnly))
        
        # Save root path 
        if browse_path != '':
            self.root_dir = browse_path #os.path.join(browse_path, '.')
            # Save image paths
            self.left       = [x for x in sorted([img for img in os.listdir(os.path.join(self.root_dir, 'left') )])]
            self.right      = [x for x in sorted([img for img in os.listdir(os.path.join(self.root_dir, 'right') )])]
            self.left_path  = [os.path.join(self.root_dir + '/left', x)  for x in self.left ]
            self.right_path = [os.path.join(self.root_dir + '/right', x) for x in self.right ]
            self.disp_path  = [os.path.join(self.root_dir + '/disp', x)  for x in self.left ]

        # Reset and show image list
        if self.left_path is not None:
            self.list_image.clear()

        for i in range(len(self.left)):
            self.list_image.insertItem(i, self.left[i])

    def show_frame_in_display(self,image_path, lbl_img):
        try:
            image_profile = QImage(image_path)
            image_profile = image_profile.scaled(self.img_width,self.img_height, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            lbl_img.setPixmap(QPixmap.fromImage(image_profile))
        except:
            pass

    def track_progress(self, i):
        # For each image pair computed, update the track bar
        self.progressBar.setValue((i+1)*100/len(self.process_left_path))


    def evaluate(self, all):
        
        # Disable all button to start computing
        self.btn_eval1.setEnabled(False)
        self.btn_evalall.setEnabled(False)
        self.progressBar.setValue(0)

        # Load image path for computing
        self.process_left_path =[]
        self.process_right_path = []

        if all == True:
            self.eval_all = True
            self.process_left_path = self.left_path
            self.process_right_path = self.right_path
            self.res_disp = self.disp_path
        else:
            self.process_left_path  = [self.left_path[self.idx]]
            self.process_right_path = [self.right_path[self.idx]]
            self.res_disp = [self.disp_path[self.idx]]

        self.label_7.setText("Working")

        # Listen to worker to process image pairs
        worker = Worker(self.process_left_path, self.process_right_path, self.res_disp, self.idx)
        worker.signals.result.connect(self.track_progress)
        worker.signals.finished.connect(self.work_finished)

        # Execute
        try:
            self.threadpool.start(worker)
        except:
            pass
    
    def work_finished(self):
        # Display the computed results
        self.display_seleted_image()
        self.label_7.setText("Finished")
        # Enable buttons
        self.btn_eval1.setEnabled(True)
        self.btn_evalall.setEnabled(True)

    def display_seleted_image(self):

        if self.idx >=0:

            self.v_disp.clear()

            left_pth = self.left_path[self.idx]
            right_pth = self.right_path[self.idx]
            disp_pth = self.disp_path[self.idx]
            file_name = os.path.basename(left_pth)

            # Display selected images in the views
            try:
                if os.path.isfile(disp_pth):
                    self.show_frame_in_display(disp_pth, self.v_disp)

                self.show_frame_in_display(left_pth, self.v_left)
                self.show_frame_in_display(right_pth, self.v_right)
            except:
                pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())