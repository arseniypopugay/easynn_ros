import sys
import os, os.path

from PIL import Image, ImageDraw
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QListWidget, QListWidgetItem, QPushButton, QInputDialog, QLineEdit, \
    QFileDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PIL.ImageQt import ImageQt
from qtpy import QtGui, QtWidgets

import TrainStarter

app = QApplication(sys.argv)

IMAGES_FOLDER_PATH = ""

image_marking_data = {
    "class_labels": [],
    "image_files": [],
    "marks": []
}


class ClassLabelsList():
    def __init__(self, next_step):
        self.stage = "label_creating"

        self.next_step = next_step
        self.class_labels = []
        self.window_labels = QWidget()
        self.window_labels.setWindowTitle('Classes labels')
        self.window_labels.setFixedSize(200, 350)

        self.listWidget = QListWidget(parent=self.window_labels)
        self.listWidget.setGeometry(0, 0, 200, 200)

        self.addClassButton = QPushButton("Add class", parent=self.window_labels)
        self.addClassButton.setGeometry(10, 200, 180, 40)
        self.addClassButton.clicked.connect(self.add_class_button_pressed)

        self.removeClassButton = QPushButton("Remove selected", parent=self.window_labels)
        self.removeClassButton.setGeometry(10, 240, 180, 40)
        self.removeClassButton.clicked.connect(self.remove_class_button_pressed)

        self.finishLabelAddingButton = QPushButton("Start images marking", parent=self.window_labels)
        self.finishLabelAddingButton.setGeometry(10, 300, 180, 40)
        self.finishLabelAddingButton.clicked.connect(self.finish_lable_adding_pressed)

    def add_class_button_pressed(self):
        text, okPressed = QInputDialog.getText(QWidget(), "New class", "Class name:", QLineEdit.Normal, "")
        if okPressed:
            if text.lower() not in self.class_labels:
                QListWidgetItem(text.lower(), self.listWidget)
                self.class_labels.append(text.lower())

    def remove_class_button_pressed(self):
        listItems = self.listWidget.selectedItems()
        if not listItems: return
        for item in listItems:
            self.class_labels.remove(item.text())
            self.listWidget.takeItem(self.listWidget.row(item))

    def finish_lable_adding_pressed(self):
        if len(self.class_labels) < 1:
            print("Must be >0 labels")
            return
        print(self.class_labels)
        IMAGES_FOLDER_PATH = str(QFileDialog.getExistingDirectory(QWidget(), "Select Directory"))

        valid_images = [".jpg", ".jpeg", ".png"]
        images_list = []
        for f in os.listdir(IMAGES_FOLDER_PATH):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            images_list.append(os.path.join(IMAGES_FOLDER_PATH, f))

        image_marking_data.update({"image_files": images_list})
        image_marking_data.update({"class_labels": self.class_labels})
        print(image_marking_data)

        self.window_labels.setFixedSize(200, 200)
        self.finishLabelAddingButton.hide()
        self.addClassButton.hide()
        self.removeClassButton.hide()
        self.next_step()

    def show(self):
        self.window_labels.show()


class ImageMarker:
    def __init__(self, classLabelsList):
        self.classLabelsList = classLabelsList
        self.window_marker = QWidget()
        self.window_marker.setWindowTitle('Image marker')
        self.window_marker.setFixedSize(800, 600)
        self.label_image_container = QLabel(parent=self.window_marker)
        self.label_image_container.mouseReleaseEvent = self.label_image_click

        self.show_prev_image_button = QPushButton("Prev", parent=self.window_marker)
        self.show_prev_image_button.setGeometry(10, 500, 100, 40)
        self.show_prev_image_button.clicked.connect(self.show_prev_image)

        self.show_next_image_button = QPushButton("Next", parent=self.window_marker)
        self.show_next_image_button.setGeometry(200, 500, 100, 40)
        self.show_next_image_button.clicked.connect(self.show_next_image)

        self.start_training_button = QPushButton("Start training", parent=self.window_marker)
        self.start_training_button.setGeometry(300, 500, 100, 40)
        self.start_training_button.clicked.connect(self.start_training)

        self.current_image_file = ""
        self.current_image_num = 0
        self.im = None
        self.last_clicked_coords = []

    def show(self):
        self.update_image()

        self.label_image_container.show()
        self.window_marker.show()

    def update_image(self):
        self.current_image_file = image_marking_data['image_files'][self.current_image_num]
        self.im = Image.open(self.current_image_file)
        w, h = self.im.size

        coef = max(w, h) / 400
        self.im = self.im.resize((int(w / coef), int(h / coef)), Image.NEAREST)
        d = ImageDraw.Draw(self.im)

        self.current_image_marks = []
        for mark in image_marking_data['marks']:
            if mark['image'] == self.current_image_file:
                self.current_image_marks.append(mark)
        w, h = self.im.size
        for mark in self.current_image_marks:
            coords = tuple(map(int, [mark['corners'][0][0] * w, mark['corners'][0][1] * h, mark['corners'][1][0] * w,
                                     mark['corners'][1][1] * h]))
            print(coords)
            d.rectangle(xy=coords, outline=(255, 0, 0))
            d.text(xy=(int((coords[0] + coords[2]) / 2), int((coords[1] + coords[3]) / 2)), text=mark['label'],
                   fill='red')

        if self.last_clicked_coords:
            clicked_x = int(self.last_clicked_coords[0])
            clicked_y = int(self.last_clicked_coords[1])
            d.ellipse(xy=((clicked_x - 1, clicked_y - 1), (clicked_x + 1, clicked_y + 1)), fill=(255, 0, 0))

        self.im = self.im.convert("RGBA")
        qim = ImageQt(self.im)
        pixmap = QtGui.QPixmap.fromImage(qim)
        self.label_image_container.setPixmap(pixmap)

    def show_prev_image(self):
        self.last_clicked_coords = []
        if self.current_image_num - 1 >= 0:
            self.current_image_num -= 1
            self.update_image()

    def show_next_image(self):
        self.last_clicked_coords = []
        if self.current_image_num + 1 < len(image_marking_data['image_files']):
            self.current_image_num += 1
            self.update_image()

    def label_image_click(self, event):
        click_x = event.x()
        click_y = event.y()
        print(click_x, click_y)

        if self.last_clicked_coords:

            selected_labels = self.classLabelsList.listWidget.selectedItems()
            if len(selected_labels) == 1:
                label = selected_labels[0].text()

                w, h = self.im.size
                left_up_corner = [min(click_x, self.last_clicked_coords[0]) / w,
                                  min(click_y, self.last_clicked_coords[1]) / h]
                right_bottom_corner = [max(click_x, self.last_clicked_coords[0]) / w,
                                       max(click_y, self.last_clicked_coords[1]) / h]
                image_marking_data['marks'].append({
                    "image": self.current_image_file,
                    "label": label,
                    "corners": [left_up_corner, right_bottom_corner],

                })

            print(image_marking_data)

            self.last_clicked_coords = []
            pass
        else:
            self.last_clicked_coords = [click_x, click_y]
        self.update_image()

    def start_training(self):
        text, okPressed = QInputDialog.getText(QWidget(), "Model path", "Output model folder:", QLineEdit.Normal, "")
        if okPressed:
            output_dir = text
            print(image_marking_data)

            self.window_marker.hide()
            self.classLabelsList.window_labels.hide()
            TrainStarter.train(image_marking_data, text)
            exit()


# window_image = QWidget()
# window_image.setWindowTitle('Image marker')
# window_image.setGeometry(0, 0, 800, 600)
#
# window_image.show()

if __name__ == "__main__":
    classLabelsList = ClassLabelsList(next_step=None)
    image_marker = ImageMarker(classLabelsList=classLabelsList)
    classLabelsList.next_step = image_marker.show
    classLabelsList.show()

    sys.exit(app.exec_())
