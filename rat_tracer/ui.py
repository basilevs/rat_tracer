from pprint import pprint
from sys import argv, path, exit
import random
import sys
from PySide6.QtCore import QObject
from PySide6.QtCore import Qt
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtMultimedia import QMediaPlayer


if __name__ == "__main__":
    app = QGuiApplication(argv)
    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()
    # Add the current directory to the import paths and load the main module.
    engine.addImportPath(path[0])
    engine.loadFromModule(".", "Main")

    if not engine.rootObjects():
        exit(-1)

    root = engine.rootObjects()[0]

    player = root.findChild(QMediaPlayer)
    player.play()
    exit_code = app.exec()
    del engine
    exit(exit_code)