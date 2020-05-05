import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import os

import maxflow
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path
import preprocess_minh
import dahu


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Interactive segmentation'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 600
        self.initUI()
        
    
    def initUI(self):
        self.resize(1000, 600)
        self.seeds = 0
        self.segmented = 1
        self.img = None
        self.ima_lab = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.ImageName = '8_08-40-06'
        self.load_image(self.ImageName + '.jpg')
        self.seed_type = 1  #annotation type
        self.list_bg = []
        self.list_fg = []
        self.bg_markers = []
        self.fg_markers = []



        segmentButton = QPushButton("Segmentation")
        segmentButton.setStyleSheet("background-color:white")
        segmentButton.clicked.connect(self.segmentation)

        StateLine = QLabel()
        StateLine.setText("Click or Drag Left Mouse Button for Foreground Annotation,and Right Mouse Button for Background.")
        palette = QPalette()
        palette.setColor(StateLine.foregroundRole(), Qt.red)
        StateLine.setPalette(palette)

        hbox = QHBoxLayout()
        hbox.addWidget(segmentButton)
        hbox.addStretch(1)        
        hbox.addWidget(StateLine)
        hbox.addStretch(1)


        self.seedLabel = QLabel()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay(self.seeds))))
        self.seedLabel.setAlignment(Qt.AlignCenter)
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag        

        self.segmentLabel = QLabel()
        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay(self.segmented))))
        self.segmentLabel.setAlignment(Qt.AlignCenter)

        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(False)
        scroll1.setWidget(self.seedLabel)
        scroll1.setAlignment(Qt.AlignCenter)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(False)
        scroll2.setWidget(self.segmentLabel)
        scroll2.setAlignment(Qt.AlignCenter)


        imagebox = QHBoxLayout()
        imagebox.addWidget(scroll1)
        imagebox.addWidget(scroll2)

        vbox = QVBoxLayout()

        vbox.addLayout(hbox)
        vbox.addLayout(imagebox)

        self.setLayout(vbox)

         

        self.setWindowTitle('Buttons')
        self.show()

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.img, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.img, 0.3, self.segment_overlay, 0.7, 0.1)


    def load_image(self, filename):
        self.img = cv2.imread(filename)
        h = len(self.img)
        w = len(self.img[0])
        self.seed_overlay = np.zeros_like(self.img)
        self.ima_lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2Lab)
        self.segment_overlay = np.zeros_like(self.img)

    def mouse_down(self, event):
        if event.button() == Qt.LeftButton:
            self.seed_type = 1
        elif event.button() == Qt.RightButton:
            self.seed_type = 0
        print(str(event.x()) + "," + str(event.y()))
        self.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.get_image_with_overlay(self.seeds))))

    def mouse_drag(self, event):
        self.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.get_image_with_overlay(self.seeds))))

    def add_seed(self, x, y, type):
        if self.img is None:
            print('Please load an image before adding seeds.')
        if type == 0:
            if not self.bg_markers.__contains__((x, y)):
                self.bg_markers.append((x, y))
                self.list_bg.append(self.ima_lab[y][x])
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
        elif type == 1:
            if not self.fg_markers.__contains__((x, y)):
                self.fg_markers.append((x, y))
                self.list_fg.append(self.ima_lab[y][x])
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)


    def eu_dis(self, v1, v2):
        return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)



    def computeEnerge(self, label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1):
        return (self.giveDataEnerge(label, fg, bg1) + self.giveSmoothEnerge(label, neighbor, lamda,  LAB_map, sigma1))

    def giveDataEnerge(self, label, fg, bg1):
        energe = 0
        h,w = fg.shape

        for x in range (h):
            for y in range (w):
                if label[x][y] == 1:
                    energe += fg[x][y]
                elif label[x][y] == 0:
                    energe += bg1[x][y]

        return energe


    def giveSmoothEnerge(self, label, neighbor, lamda,  LAB_map, sigma1 ):  # compute SmoothEnerge
        energe = 0
        h,w = label.shape
        for x in range (h):
            for y in range (w):
                u = x*w + y
                for i in range (4):
                    a = x + neighbor[i][0]
                    b = y + neighbor[i][1]
                    if (a >= 0 and a <h and b >= 0 and b < w):
                        v = a*w + b
                        if v < u:
                            continue
                        if label[x][y] == label[a][b]:
                            continue
                        energe += lamda * np.e ** (-(self.eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
        return energe

    @pyqtSlot()
    def segmentation(self):

        h = len(self.img)
        w = len(self.img[0])

        list_bg = np.asarray(self.list_bg)
        list_fg = np.asarray(self.list_fg)

        bg_markers = np.array(self.bg_markers)
        fg_markers = np.array(self.fg_markers)

        ###### LAB map
        LAB_map_raw = cv2.cvtColor(self.img, cv2.COLOR_RGB2Lab)
        LAB_map = np.zeros_like(LAB_map_raw, dtype=np.int8)
        for i in range(len(LAB_map)):
            for j in range(len(LAB_map[0])):
                LAB_map[i, j][0] = LAB_map_raw[i, j][0] / 255 * 100
                LAB_map[i, j][1] = LAB_map_raw[i, j][1] - 128
                LAB_map[i, j][2] = LAB_map_raw[i, j][2] - 128



        score, score1 = preprocess_minh.preprocess_logpro(self.ima_lab, self.list_fg, self.list_bg)
        print("log of probability")




        # background

        # score = np.pad(score, 1, 'constant', constant_values=m0)
        score = np.array(score * 255, dtype="uint8") # convert to uint8
        #score = cv2.bilateralFilter(score,9,75,75)
        app = dahu.DahuApplication(score)
        app.setMarkers(self.fg_markers, self.bg_markers)
        fg = np.array(app.fg, dtype="float32")
        bg = np.array(app.bg, dtype="float32")

        # foreground

        # score1 = np.pad(score1, 1, 'constant', constant_values=m1)
        score1 = np.array(score1 * 255, dtype="uint8") # convert to uint8
        #score1 = cv2.bilateralFilter(score1,9,75,75)
        app1 = dahu.DahuApplication(score1)
        app1.setMarkers(self.fg_markers, self.bg_markers)
        fg1 = np.array(app1.fg, dtype="float32")
        bg1 = np.array(app1.bg, dtype="float32")


        fg = fg/255
        bg1 = bg1/255

        #### compute sigma

        neighbor = [[0,1],[0,-1],[1,0],[-1,0]]

        sigma1 = 0

        for x in range (h):
            for y in range (w):

                u = x*w + y
                for i in range (4):
                    a = x + neighbor[i][0]
                    b = y + neighbor[i][1]

                    if (a >= 0 and a <h and b >= 0 and b < w):
                        v = a*w + b

                        if (v > u):
                            if sigma1 < self.eu_dis(LAB_map[x][y], LAB_map[a][b]):
                                sigma1 = self.eu_dis(LAB_map[x][y], LAB_map[a][b])

        sigma1 = sigma1 ** 2 * 1

        print("sigma1 = " + str(sigma1))


        ############ Graphcut

        lamda  = 0.2


        label = np.zeros((h,w))


        label[fg<=bg1] = 1

        oldEnergy = self.computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
        print(oldEnergy)


        nodes = []
        edges = []

        cap_source = fg
        cap_sink = bg1

        for x in range (h):
            for y in range (w):
                u = x*w + y
                nodes.append((u, cap_source[x][y] , cap_sink[x][y]))

        for x in range (h):
            for y in range (w):
                u = x*w + y        
                for i in range (4):
                    a = x + neighbor[i][0]
                    b = y + neighbor[i][1]
                    if (a >= 0 and a <h and b >= 0 and b < w):
                        v = a*w + b
                        if (v > u):
                            weight = lamda * np.e ** (-(self.eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
                            edges.append((u, v, weight))    


        ####GraphCuts####
        g = maxflow.Graph[float](len(nodes), len(edges))

        nodelist = g.add_nodes(len(nodes))
        for node in nodes:
            g.add_tedge(node[0], node[1], node[2])

        for edge in edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()  

        for vect in nodes:
            v = vect[0]
            if g.get_segment(v) == 0:  # beta
                x = int(np.floor(v/w))
                y = v%w
                label[x][y] = 0
            else:  # alpha
                label[x][y] = 1


        newEnergy = self.computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
        print(newEnergy)

        #  Post processing

        label_gray = np.zeros(label.shape, np.uint8)

        for i in range (len(fg_markers)):
            label_gray[fg_markers[i][1], fg_markers[i][0]] = 255

        label = np.array(label, dtype="uint8") # convert to uint8
        # Find largest contour in intermediate image so that it contains the markers 
        cnts, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        con = []
        for i in range(len(cnts)):
            biggest = np.zeros(label.shape, np.uint8)
            cv2.drawContours(biggest, [cnts[i]], -1, 255, cv2.FILLED)
            biggest = cv2.bitwise_and(label_gray, biggest)
            if (np.sum(biggest) > 0):
                con.append(cnts[i])


        biggest = np.zeros(label.shape, np.uint8)

        if (len(con) != 0):
            cnt = max(con, key=cv2.contourArea)

            # Output            
            cv2.drawContours(biggest, [cnt], -1, 255, cv2.FILLED)


        biggest = np.array(biggest, dtype="uint8") # convert to uint8

        for i in range (h):
            for j in range (w):
                if biggest[i][j] == 255:
                    self.segment_overlay[i][j] = (255,255,255)

        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay(self.segmented))))


if __name__ == '__main__':
    seg = QApplication(sys.argv)
    ex = App()
    sys.exit(seg.exec_())