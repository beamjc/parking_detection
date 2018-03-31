import cv2
import numpy as np
import sys
from PyQt5 import QtGui, uic, QtWidgets, QtCore, QtWidgets, QtMultimedia
from PyQt5.QtWidgets import QFileDialog, QLabel, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal

totalparkingspot = 14
available = 0

class Form(QtWidgets.QMainWindow):
    def __init__(self):
        super(Form, self).__init__()
 
        self.ui = uic.loadUi("interface2.ui",self)
        self.startbutton.clicked.connect(self.detect_car)
    
    def detect_car(self):
        
        contourr = ''

        count = 0
        count2 = 1
        vidcap = cv2.VideoCapture('detectcar.mp4')
        while True:
            ret, image = vidcap.read()
            vidcap.set(cv2.CAP_PROP_POS_MSEC, count)
            count_car = 0


            #img = cv2.imread('frame/frame' + str(count2) + '.jpg')
            Z = image.reshape((-1,3))

            # convert to np.float32
            Z = np.float32(Z)

            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((image.shape))
            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

            ret, th1 = cv2.threshold(res2, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,3),np.uint8)
            kernel2 = np.ones((5,3),np.uint8)
            erosion = cv2.erode(th1,kernel,iterations = 7)
            dilation = cv2.dilate(erosion,kernel2,iterations = 12)
            
            contour = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[1]
            for c in contour:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("cX = " +str(cX))
                print("cY = " +str(cY))
        
                if cY <= 460:
                    count_car = count_car + 1

            image = cv2.resize(image, (800,600))
            cv2.imwrite('frame/frame' + str(count2) + '.jpg', image)
            cv2.imshow("video", image)
            count2 = count2 + 1
            count = count+10000
            self.textEdit.setText("14")
            available = totalparkingspot - count_car
            self.textEdit_3.setText(str(available))
            self.textEdit_2.setText(str(count_car))
            print(available)
            contourr = (len(contour))
#            self.textEdit_2.setText(str(contourr))
            print(len(contour))
            cv2.waitKey(30)
            
        cv2.waitKey(0)
        vidcap.release()
            
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form() 
    w.show()
    sys.exit(app.exec_())



