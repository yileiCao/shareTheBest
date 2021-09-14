import sys
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PIL.ImageQt import ImageQt
from PIL import Image
import numpy as np


class Crop_frame (QLabel):
    def __init__(self, parentQWidget:None):
        super(Crop_frame, self).__init__(parentQWidget)
        self.width = 500
        self.height = 500
        self.setObjectName("label")
        self.setScaledContents(True)
        self.resize(self.width, self.height)
        self.frame = None
        self.start_crop = False
        self.finish_crop = True
        self.rubberBand = None


    def setFrame(self, frame):
        self.frame = frame

    def set_start_crop(self, bool):
        self.start_crop = bool

    def get_start_crop(self):
        return self.start_crop

    def get_crop_geometry(self):
        return self.rubberBand.geometry().getCoords()

    def delete_rubberBand(self):
        if self.rubberBand is not None:
            self.rubberBand.deleteLater()
        self.finish_crop = True
        self.rubberBand = None

    def paintEvent(self, event):
        # draw
        if self.frame is not None:
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.frame)



    def mousePressEvent(self, ev):
        if self.frame is not None and self.start_crop and self.finish_crop:
            if ev.buttons() & Qt.LeftButton:


                self.originQPoint = ev.pos()
                print(self.originQPoint)
                self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.originQPoint, QSize()))
                self.rubberBand.show()

    def mouseMoveEvent(self, ev):
        if self.frame is not None and self.start_crop and self.finish_crop:
            if ev.buttons() & Qt.LeftButton:
                self.rubberBand.setGeometry(QRect(self.originQPoint, ev.pos()).normalized())

    def mouseReleaseEvent(self, ev):
        if self.frame is not None and self.start_crop and self.finish_crop:
            self.finish_crop = False
            print(self.rubberBand.geometry())
            print(self.rubberBand.geometry().getCoords()[0])


    def image2pixmap(self, image, size):
        desired_size = size
        im = Image.fromarray(image).convert('RGB')
        old_size = im.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))
        pixmap = QPixmap.fromImage(ImageQt(new_im).copy())
        return pixmap

        '''
        if ev.buttons() & Qt.LeftButton:
            print("ee")
            self.rubberBand.hide()
            currentQRect = self.rubberBand.geometry()
            self.rubberBand.deleteLater()

            pen = QPen(Qt.green, 5)

            painter = QPainter(self)
            painter.setPen(pen)
            painter.drawPixmap(self.rect(), self.image)
            painter.drawRect(currentQRect)
            print("ee")
            painter.drawEllipse(300, 300, 200, 200)
        '''

    '''
    def paintEvent(self, event):
        if self.mModified == True:
            qp = QPainter()
            qp.begin(self)
            qp.setPen(QPen(Qt.green, 8, Qt.DashLine))
            qp.drawRect(self.rubberBand.geometry())
            qp.end()
            self.mModified = False
            self.rubberBand.deleteLater()


    def mousePressEvent(self, event):
        self.rubberBand = None
        self.originQPoint = event.pos()
        print(self.originQPoint)

        if not self.rubberBand:
            self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
            self.rubberBand.setGeometry(QRect(self.originQPoint, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, eventQMouseEvent):
        self.rubberBand.setGeometry(QRect(self.originQPoint, eventQMouseEvent.pos()).normalized())

    def mouseReleaseEvent(self, eventQMouseEvent):
        self.rubberBand.hide()
        currentQRect = self.rubberBand.geometry()
        print(currentQRect)
        print(currentQRect.getRect())
        # self.rubberBand.deleteLater()
        cropQPixmap = self.pixmap().copy(currentQRect)
        print(cropQPixmap.size())
        # self.raw_frame_2.setPixmap(cropQPixmap)
        self.mModified = True
    '''

if __name__ == '__main__':
    myQApplication = QApplication(sys.argv)
    image_path = "/Users/yileicao/Documents/Birmingham_project/summer_project/ppt_newn/5.828_1.528.jpg"
    image = Image.open(image_path).convert("RGB")
    myQExampleLabel = Crop_frame(None)
    pixmap = myQExampleLabel.image2pixmap(np.array(image), 450)
    myQExampleLabel.setFrame(pixmap)
    myQExampleLabel.set_start_crop(True)
    myQExampleLabel.show()
    sys.exit(myQApplication.exec())
