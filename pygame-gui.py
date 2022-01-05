from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QMainWindow, QStatusBar
import pygame
import sys
from pygame.locals import *
#get GUi settings
windowWidth = 500
windowHeight = 500
#colors
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
black = (0,0,0)
yellow = (240,240,0)
class ImageWidget(QWidget):
    def __init__(self,surface,parent=None):
        super(ImageWidget,self).__init__(parent)
        w=surface.get_width()
        h=surface.get_height()
        self.data=surface.get_buffer().raw
        self.image=QtGui.QImage(self.data,w,h,QtGui.QImage.Format_RGB32)
    def paintEvent(self,event):
        qp=QtGui.QPainter()
        qp.begin(self)
        qp.drawImage(0,0,self.image)
        qp.end()
class MainWindow(QMainWindow):
    global windowWidth
    global windowHeight
    def __init__(self,surface,parent=None):
        global windowWidth
        global windowHeight
        super(MainWindow,self).__init__(parent)
        self.setGeometry(500, 100, windowWidth, windowHeight)
        #self.move(QApplication.desktop().screen().rect().center() - self.rect().center())
        self.setCentralWidget(ImageWidget(surface))
        #self.showFullScreen()
        #screen_resolution = app.desktop().screenGeometry()
        #windowWidth, windowHeight = screen_resolution.width(), screen_resolution.height()
        self.setGeometry(0, 0, windowWidth, windowHeight)
        self.createUI()
    def createUI(self):
        self.setWindowTitle('Test game')
        #Menubar
        #First item
        menu = self.menuBar().addMenu('Menu')
        menuSettings = menu.addAction('Settings')
        #action.triggered.connect(self.changeFilePath)
        menuNewGame = menu.addAction('New Game')
        menuSaveGame = menu.addAction('Save Game')
        menuLoadGame = menu.addAction('Load Game')
        # Statusbar
        #self.statusBar().showMessage("test",5000)
        self.statusBar().showMessage("Game Loaded")

pygame.init()

gameScreen =pygame.Surface((400,200))
gameScreen.fill(red)
pygame.draw.circle(gameScreen,(255,255,255,255),(100,100),50)
app=QApplication(sys.argv)
mainWindow=MainWindow(gameScreen)
mainWindow.show()
app.exec_()