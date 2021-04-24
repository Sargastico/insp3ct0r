from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1162, 602)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.result_view_frame = QtWidgets.QLabel(self.centralwidget)
        self.result_view_frame.setGeometry(QtCore.QRect(590, 60, 561, 401))
        self.result_view_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.result_view_frame.setLineWidth(1)
        self.result_view_frame.setText("")
        self.result_view_frame.setObjectName("result_view_frame")
        self.camera_view_label = QtWidgets.QLabel(self.centralwidget)
        self.camera_view_label.setGeometry(QtCore.QRect(10, 30, 561, 20))
        self.camera_view_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_view_label.setObjectName("camera_view_label")
        self.camera_view_frame = QtWidgets.QLabel(self.centralwidget)
        self.camera_view_frame.setGeometry(QtCore.QRect(10, 60, 561, 401))
        self.camera_view_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.camera_view_frame.setLineWidth(1)
        self.camera_view_frame.setText("")
        self.camera_view_frame.setObjectName("camera_view_frame")
        self.result_view_label = QtWidgets.QLabel(self.centralwidget)
        self.result_view_label.setGeometry(QtCore.QRect(590, 30, 561, 20))
        self.result_view_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_view_label.setObjectName("result_view_label")
        self.CaptureButton = QtWidgets.QPushButton(self.centralwidget)
        self.CaptureButton.setGeometry(QtCore.QRect(10, 480, 89, 25))
        self.CaptureButton.setObjectName("CaptureButton")
        # MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera_view_label.setText(_translate("MainWindow", "CAMERA VIEW"))
        self.result_view_label.setText(_translate("MainWindow", "RESULT VIEW"))
        self.CaptureButton.setText(_translate("MainWindow", "Capture"))
