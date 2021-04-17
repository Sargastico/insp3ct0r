# import system module
import sys

from PyQt5.QtWidgets import *
from ui.main_window_events import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    mainWindow.showMaximized()

    sys.exit(app.exec_())
