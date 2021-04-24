import cv2

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSlot

from flow.processing_flow import processFrame
from ui.main_window_ui import Ui_MainWindow

class MainWindow(QWidget):

    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.setFixedSize(1162, 602)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.controlTimer()
        self.initUI()

    def initUI(self):

        self.ui.CaptureButton.clicked.connect(self.CaptureEvent)
        # button = QPushButton('CaptureButton', self)
        # button.clicked.connect(self.CaptureEvent)
        self.show()

    @pyqtSlot()
    def CaptureEvent(self):

        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processFrame(image, 'assets/baseimage.jpg')
        self.ui.result_view_frame.setPixmap(QPixmap('./assets/result.jpg'))

    # view camera
    def viewCam(self):

        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.camera_view_frame.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):

        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text

        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()