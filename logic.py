from mainwindow import Ui_MainWindow
from PySide6 import QtCore, QtGui, QtWidgets
from PyQt_function.tools import *
from PIL.ImageQt import ImageQt
from PIL import Image, ImageOps


class Logic(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.resize(1024, 570)

        self.mModified = False
        self.undo_history = None
        self.redo_history = None
        self.result = None
        self.enhanced_result = None
        self.view_images = None
        self.select_frame = None
        self.crop_raw_frame = None
        self.num_frame = 100

        # Set connection
        self.horizontalSlider.valueChanged.connect(self.v_change)
        self.forward_pushButton.clicked.connect(lambda: self.forward_pushButton_click())
        self.afterward_pushButton.clicked.connect(lambda: self.afterward_pushButton_click())
        self.goto_pushButton.clicked.connect(lambda: self.gotoNum_pushButton_click())
        self.pushButton_A.clicked.connect(lambda: self.save_pushButtons_click("A"))
        self.pushButton_B.clicked.connect(lambda: self.save_pushButtons_click("B"))
        self.pushButton_C.clicked.connect(lambda: self.save_pushButtons_click("C"))

        self.pushButton_gotoA.clicked.connect(lambda: self.goto_pushButtons_click(0))
        self.pushButton_gotob.clicked.connect(lambda: self.goto_pushButtons_click(1))
        self.pushButton_gotoc.clicked.connect(lambda: self.goto_pushButtons_click(2))

        self.pushButton_enhance.clicked.connect(lambda: self.pushButton_enhance_click())

        self.raw_radioButton.clicked.connect(lambda: self.tab2_radio_state(self.raw_radioButton))
        self.enhanced_radioButton.clicked.connect(lambda: self.tab2_radio_state(self.enhanced_radioButton))
        self.raw_radioButton.setChecked(True)
        self.undo_pushButton.clicked.connect(lambda: self.undo_pushButton_tab1_click())
        self.redo_pushButton.clicked.connect(lambda: self.redo_pushButton_tab1_click())
        self.radioButton_1.clicked.connect(lambda: self.tab1_radio_state(0))
        self.radioButton_2.clicked.connect(lambda: self.tab1_radio_state(1))
        self.radioButton_3.clicked.connect(lambda: self.tab1_radio_state(2))
        self.radioButton_1.setChecked(True)
        self.cropButton.clicked.connect(lambda: self.cropButton_click())
        self.confirmButton.clicked.connect(lambda: self.confirmButton_click())

        self.save_raw_rb.clicked.connect(lambda: self.select_save_image(0))
        self.save_enhance_rb.clicked.connect(lambda: self.select_save_image(1))
        self.save_pushButton.clicked.connect(lambda: self.save_image())

        # set up models
        self.nima = set_up_nima()
        self.lut0, self.lut1, self.lut2, self.lut_classifier, self.lut_trilinear = set_up_lut()

    def openCall(self):
        filepath = self.open_dialog_box()
        self.statusbar.showMessage("Scoring images...")
        self.result = video_result(self.nima, filepath)
        self.set_up_gui()

    def openFolderCall(self):
        filepath = self.open_folder_dialog_box()
        self.statusbar.showMessage("Scoring images...")
        self.result = images_result(self.nima, filepath)
        self.set_up_gui()

    def set_up_gui(self):
        index = sorted(range(len(self.result)), key=lambda l: self.result[l][1], reverse=True)
        position_list = ['A', 'B', 'C']
        im_type = ['Raw image', 'Raw image', 'Raw image']
        # no close images(20 frames)
        top3index = self.top3(index)
        top3images = [self.result[top3index[0]]] + [self.result[top3index[1]]] + [self.result[top3index[2]]]
        self.view_images = list(zip(position_list, top3images, top3index, im_type))
        # [('A',(image, mean, std), position, 'Raw image'),
        # ('B',(image, mean, std), position, 'Raw image'), ('C',(image, mean, std), position, 'Raw image')]

        # set up tab 1
        image_1_pixmap = self.image2pixmap(self.result[top3index[0]][0], 300)
        image_2_pixmap = self.image2pixmap(self.result[top3index[1]][0], 300)
        image_3_pixmap = self.image2pixmap(self.result[top3index[2]][0], 300)
        self.image_1.setPixmap(image_1_pixmap)
        self.image_2.setPixmap(image_2_pixmap)
        self.image_3.setPixmap(image_3_pixmap)
        self.text_1.setText(self.image_text_position(self.result[top3index[0]][1], self.result[top3index[0]][2], top3index[0]))
        self.text_2.setText(self.image_text_position(self.result[top3index[1]][1], self.result[top3index[1]][2], top3index[1]))
        self.text_3.setText(self.image_text_position(self.result[top3index[2]][1], self.result[top3index[2]][2], top3index[2]))

        # set up tab 2(raw_image part)
        self.select_frame = self.result[top3index[0]] + (top3index[0], "Raw image")
        big_frame_pixmap = self.image2pixmap(self.select_frame[0], 450)
        self.num_frame = len(self.result)
        self.big_frame.setPixmap(big_frame_pixmap)
        self.video_information_label.setText(
            f'''
        This video includes {self.num_frame} images,
        Image resolution is {self.result[0][0].shape[0]} X {self.result[0][0].shape[1]},
        The highest score is {self.result[index[0]][1]:.2f}
                    ''')

        # enhance video/images
        self.statusbar.showMessage("Ready...")
        # self.enhanced_result = enhancement_result(self.result, self.lut0, self.lut1, self.lut2,
        #                                           self.lut_classifier, self.lut_trilinear, self.nima)

        # set up horizontal slider
        self.horizontalSlider.setMaximum(self.num_frame)
        self.horizontalSlider.setValue(top3index[0])
        self.v_change()
        self.tab1_radio_state(0)

    def top3(self, index, distance=10):
        """
        Args:
            index: index of sorted frames by aesthetic score
            distance: two frames less than this distance are considered similar images

        Returns: Top 3 index where images are dissimilar
        """
        result = []
        pos = 0
        maxi = len(index)
        similar_dir = {index[0],}
        result.append(index[0])
        for iter in range(2):
            for i in range(index[pos]-distance, index[pos]+distance+1):
                if 0 <= i < maxi:
                    similar_dir.add(i)
            while pos < maxi-1 and index[pos] in similar_dir:
                pos += 1
            result.append(index[pos])
        return result

    def forward_pushButton_click(self):
        slider_value = self.horizontalSlider.value()
        self.horizontalSlider.setValue(slider_value - 1)

    def afterward_pushButton_click(self):
        slider_value = self.horizontalSlider.value()
        self.horizontalSlider.setValue(slider_value + 1)

    def gotoNum_pushButton_click(self):
        lineEdit_value = self.lineEdit.text()
        if lineEdit_value.isdigit():
            value = int(lineEdit_value)
            if 1 <= value <= self.num_frame:
                self.horizontalSlider.setValue(value)

    def goto_pushButtons_click(self, position):
        if self.view_images:
            self.lineEdit.setText(str(self.view_images[position][2]))
            self.gotoNum_pushButton_click()

    def undo_pushButton_tab1_click(self):
        if self.undo_history is not None:
            if self.undo_history[0] == 'A':
                frame_pixmap = self.image2pixmap(self.undo_history[1][0], 450)
                self.image_1.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.undo_history[1][1:], *self.undo_history[2:])
                self.text_1.setText(text)
                self.redo_history = self.view_images[0]
                self.view_images[0] = self.undo_history
                self.undo_history = None
            elif self.undo_history[0] == 'B':
                frame_pixmap = self.image2pixmap(self.undo_history[1][0], 450)
                self.image_2.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.undo_history[1][1:], *self.undo_history[2:])
                self.text_2.setText(text)
                self.redo_history = self.view_images[1]
                self.view_images[1] = self.undo_history
                self.undo_history = None
            elif self.undo_history[0] == 'C':
                frame_pixmap = self.image2pixmap(self.undo_history[1][0], 450)
                self.image_3.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.undo_history[1][1:], *self.undo_history[2:])
                self.text_3.setText(text)
                self.redo_history = self.view_images[2]
                self.view_images[2] = self.undo_history
                self.undo_history = None

    def redo_pushButton_tab1_click(self):
        if self.redo_history is not None:
            if self.redo_history[0] == 'A':
                frame_pixmap = self.image2pixmap(self.redo_history[1][0], 450)
                self.image_1.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.redo_history[1][1:], *self.redo_history[2:])
                self.text_1.setText(text)
                self.undo_history = self.view_images[0]
                self.view_images[0] = self.redo_history
                self.redo_history = None
            elif self.redo_history[0] == 'B':
                frame_pixmap = self.image2pixmap(self.redo_history[1][0], 450)
                self.image_2.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.redo_history[1][1:], *self.redo_history[2:])
                self.text_2.setText(text)
                self.undo_history = self.view_images[1]
                self.view_images[1] = self.redo_history
                self.redo_history = None
            elif self.redo_history[0] == 'C':
                frame_pixmap = self.image2pixmap(self.redo_history[1][0], 450)
                self.image_3.setPixmap(frame_pixmap)
                text = self.image_text_position(*self.redo_history[1][1:], *self.redo_history[2:])
                self.text_3.setText(text)
                self.undo_history = self.view_images[2]
                self.view_images[2] = self.redo_history
                self.redo_history = None

    def save_pushButtons_click(self, position):

        if position == "A" and self.view_images:
            self.undo_history = self.view_images[0]
            # ('A', (image, mean, std), position, im_type)
            self.view_images[0] = ('A',  self.select_frame[:3]) + self.select_frame[3:]
            frame_pixmap = self.image2pixmap(self.select_frame[0], 450)
            self.image_1.setPixmap(frame_pixmap)
            text = self.image_text_position(*self.select_frame[1:])
            self.text_1.setText(text)
        if position == "B" and self.view_images:
            self.undo_history = self.view_images[1]
            self.view_images[1] = ('B', self.select_frame[:3]) + self.select_frame[3:]
            frame_pixmap = self.image2pixmap(self.select_frame[0], 450)
            self.image_2.setPixmap(frame_pixmap)
            text = self.image_text_position(*self.select_frame[1:])
            self.text_2.setText(text)
        if position == "C" and self.view_images:
            self.undo_history = self.view_images[2]
            self.view_images[2] = ('C', self.select_frame[:3]) + self.select_frame[3:]
            frame_pixmap = self.image2pixmap(self.select_frame[0], 450)
            self.image_3.setPixmap(frame_pixmap)
            text = self.image_text_position(*self.select_frame[1:])
            self.text_3.setText(text)

    def pushButton_enhance_click(self):
        slider_value = self.horizontalSlider.value()
        raw_image = self.result[slider_value][0]
        self.enhanced_result = frame_enhance(raw_image, self.lut0, self.lut1,
                                                                  self.lut2,self.lut_classifier,
                                                                  self.lut_trilinear, self.nima)
        raw_image_pixmap = self.image2pixmap(raw_image, 230)
        self.raw_frame.setPixmap(raw_image_pixmap)
        self.raw_frame_label.setText(self.image_text(
            self.result[slider_value][1], self.result[slider_value][2]))

        enhanced_image_pixmap = self.image2pixmap(self.enhanced_result[0], 230)
        self.enhanced_frame.setPixmap(enhanced_image_pixmap)
        self.enhanced_frame_label.setText(self.image_text(self.enhanced_result[1], self.enhanced_result[2]))

    def v_change(self):
        if self.view_images:
            slider_value = self.horizontalSlider.value()
            self.lineEdit.setText(str(slider_value))
            big_frame_pixmap = self.image2pixmap(
                self.result[slider_value][0], 450)
            self.big_frame.setPixmap(big_frame_pixmap)

    def tab2_radio_state(self, b):
        if b.isChecked() and self.select_frame:
            if b.text() == "Raw":
                self.select_frame = self.result[self.horizontalSlider.value()] + (self.horizontalSlider.value(), "Raw image")
            elif b.text() == "Enhanced":
                self.select_frame = self.enhanced_result + (self.horizontalSlider.value(), "Enhanced image")
            big_frame_pixmap = self.image2pixmap(
                self.select_frame[0], 450)
            self.big_frame.setPixmap(big_frame_pixmap)

    def tab1_radio_state(self, position):
        if self.view_images:
            self.crop_raw_frame = self.view_images[position][1]

            crop_frame_pixmap = self.image2pixmap(
                self.crop_raw_frame[0], 450)
            self.crop_frame.setFrame(crop_frame_pixmap)
            self.crop_frame_label.setText(self.image_text_position2(self.view_images[position]))

    def cropButton_click(self):
        self.crop_frame.set_start_crop(True)
        self.crop_frame.delete_rubberBand()

    def confirmButton_click(self):
        crop_image = None
        if self.crop_frame.get_start_crop() and self.crop_raw_frame:
            x, y, x_, y_ = self.crop_frame.get_crop_geometry()
            width = x_ - x
            height = y_ - y
            frame_x, frame_y, frame_w, frame_h = 30, 10, 450, 450
            raw_h, raw_w, _ = self.crop_raw_frame[0].shape
            ratio = max(raw_w, raw_h) / 450
            if raw_h >= raw_w:
                x_index1 = int(ratio * x - (450 * ratio - raw_w) / 2)
                x_index2 = int(x_index1 + (width * ratio))
                y_index1 = int(ratio * y)
                y_index2 = int(y_index1 + height * ratio)
            else:
                x_index1 = int(ratio * x)
                x_index2 = int(x_index1 + width * ratio)
                y_index1 = int(ratio * y - (450 * ratio - raw_h) / 2)
                y_index2 = int(y_index1 + height * ratio)

            x_index1 = max(0, x_index1)
            x_index1 = min(x_index1, raw_w)
            x_index2 = max(0, x_index2)
            x_index2 = min(x_index2, raw_w)
            y_index1 = max(0, y_index1)
            y_index1 = min(y_index1, raw_h)
            y_index2 = max(0, y_index2)
            y_index2 = min(y_index2, raw_h)

            crop_image = self.crop_raw_frame[0][y_index1:y_index2, x_index1:x_index2, :]
        elif not self.crop_frame.get_start_crop() and self.crop_raw_frame:
            crop_image = self.crop_raw_frame[0]

        if crop_image is not None:
            self.raw_image, raw_mu, raw_std = frame_eva(self.nima, crop_image)
            self.raw_frame_2.setPixmap(self.image2pixmap(self.raw_image, 230))
            self.raw_frame_label_2.setText(self.image_text(raw_mu, raw_std))

            self.enhanced_image, enhanced_mu, enhanced_std = frame_enhance(self.raw_image, self.lut0, self.lut1, self.lut2,
                                                self.lut_classifier, self.lut_trilinear, self.nima)

            self.enhanced_frame_2.setPixmap(self.image2pixmap(self.enhanced_image, 230))
            self.enhanced_frame_label_2.setText(self.image_text(enhanced_mu, enhanced_std))

            self.crop_frame.set_start_crop(False)
            self.crop_frame.delete_rubberBand()

    def select_save_image(self, enhance):
        self.image_to_save = None
        if enhance and self.view_images:
            self.image_to_save = self.enhanced_image
        elif not enhance and self.view_images:
            self.image_to_save = self.raw_image
        if self.image_to_save is not None:
            self.image_to_save = Image.fromarray(self.image_to_save.astype(np.uint8))
            # change here
            # im_size = self.image_to_save.size
            # desired_size = max(im_size)
            # delta_w = desired_size - im_size[0]
            # delta_h = desired_size - im_size[1]
            # padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            # self.image_to_save = ImageOps.expand(self.image_to_save, padding)

    def save_image(self):
        filePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        # if file path is blank return back
        if filePath == "":
            return
        # saving canvas at desired path
        self.image_to_save.save(filePath)

    def open_folder_dialog_box(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        return directory

    def open_dialog_box(self):
        filename = QtWidgets.QFileDialog.getOpenFileName()
        return filename[0]

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
        pixmap = QtGui.QPixmap.fromImage(ImageQt(new_im).copy())
        return pixmap

    def image_text(self, mean, std):
        text = f"Score:     {mean:.2f} \nVariance: {std:.2f}"
        return text

    def image_text_position(self, mean, std, position, im_type='Raw image'):
        text = f"{im_type} \nScore:     {mean:.2f} \nVariance: {std:.2f} \nPosition:  {position}"
        return text

    def image_text_position2(self, view_image):
        mean = view_image[1][1]
        std = view_image[1][2]
        position = view_image[2]
        im_type = view_image[3]
        return self.image_text_position(mean, std, position, im_type)


    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(
            self,
            'Quit?',
            'Are you sure you want to quit ?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            event.accept()
        else: event.ignore()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Logic(None)
    ui.showMaximized()
    sys.exit(app.exec())



