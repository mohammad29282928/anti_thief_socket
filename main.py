import cv2
import jdatetime 
import os
import glob
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sqlite3
from datetime import date
from PyQt5.QtPrintSupport import QPrinter
import time

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import subprocess 
import psutil
import pickle
import multiprocessing



from tracer import detect
from database import making_object_db, making_logs_db
from take_screenshot import take_screen


class DrawLineWidget(object):
    def __init__(self):
        self.original_image = cv2.imread('screen.jpg')
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)
        # List to store start/end points
        self.image_coordinates = []
        self.satrt = ()
        self.end = ()

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            
        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            self.start = self.image_coordinates[0]
            self.end   = self.image_coordinates[1]

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 


        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

def run_draw():
    draw_line_widget = DrawLineWidget()
    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    return draw_line_widget.start, draw_line_widget.end

def draw_boxes():

    global ix, iy, drawing, img, boxes
    img = cv2.imread("screen.jpg")
    #img2 = cv2.imread("flower.jpg")

    # variables
    ix = -1
    iy = -1
    drawing = False
    boxes= []

    def draw_reactangle_with_drag(event, x, y, flags, param):
        global ix, iy, drawing, img, boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix = x
            iy = y


        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img2 = cv2.imread("screen.jpg")
                cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=1)
                img = img2
                boxes = [(ix,iy), (x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            img2 = cv2.imread("screen.jpg")
            cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=1)
            img = img2

    cv2.namedWindow(winname= "Title of Popup Window")
    cv2.setMouseCallback("Title of Popup Window", draw_reactangle_with_drag)
    
    while True:
        cv2.imshow("Title of Popup Window", img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    return boxes[0], boxes[1]

def initiat_data_base():
    files = glob.glob('databases/*')
    for f in files:
        os.remove(f)
            
def check_connnections(cam_id):
    if str(cam_id).isdigit():
        cam_id = int(cam_id)
    #Check cam
    cam1_ok = False    
    cam1 = cv2.VideoCapture(cam_id) 
    if cam1.isOpened():
        cam1_ok = True
    return cam1_ok
  

ui, _ = loadUiType('page.ui')

class MainApp(QMainWindow,ui):

    def __init__(self):
        QMainWindow.__init__(self)
        
        self.setupUi(self)
        self.cordinate = [(0, 0), (0, 0)]  
        self.cameras = []
        
        self.detection_object_type.addItems([item for item in ['person', "bag_rice", "car", "cap", "fire", "fire hydrant"]])
        self.box_object_type.addItems([item for item in ['truck', 'use_draw_box']])
        self.alarm_type.addItems([item for item in ['None', 'email', "sms", 'audio', "all"]])
        
        self.draw_new_box.clicked.connect(self.DrawBox)
        self.run_counter.clicked.connect(self.RunCounter)
        self.stop_counter.clicked.connect(self.StopCounter)
        self.clear_objects_db.clicked.connect(self.clear_objects_db_service)
        self.add_camera.clicked.connect(self.add_camera_service)
        self.remove_camera.clicked.connect(self.remove_camera_service)
        self.show_results.clicked.connect(self.show_results_service)
        
        self.proc = 0
        self.process = []
        initiat_data_base() # removing all databases

        
    def show_results_service(self):
        cam_name = str(self.list_of_cameras.currentText())
        if os.path.exists(os.path.join('.', 'databases', f'{cam_name}.db')):
            conn = sqlite3.connect(os.path.join('.', 'databases', f'{cam_name}.db'))
            cursor = conn.cursor()
            cursor.execute("""SELECT * from objects""")
            records = cursor.fetchall()
            conn.close()
        else:
            records = []
        
        if len(records):
            object_in = int(records[-1][2])
            object_out = int(records[-1][3])
        else:
            object_in = 0
            object_out = 0

        self.object_in.setText(str(object_in))
        self.object_out.setText(str(object_out))
                       
    def remove_camera_service(self):
        removed_item = str(self.list_of_cameras.currentText())
        self.list_of_cameras.clear()
        cameras_copy =[]
        for item in self.cameras:
            if item["camera_name"] != removed_item:
                cameras_copy.append(item)
                
        self.cameras = cameras_copy
        self.list_of_cameras.addItems([elem['camera_name'] for elem in cameras_copy])
        self.message_box.setText(f"A new camera by the name {removed_item } deleted")
     
    def add_camera_service(self):
        item = {
            "source": self.camera_ip.text(),
            "start_point": str(self.cordinate[0]),
            "end_point": str(self.cordinate[1]),
            "detection_object_type": str(self.detection_object_type.currentText()),
            "box_object_type": str(self.box_object_type.currentText()),
            "alarm_type": str(self.alarm_type.currentText()),
            "phone_numbers": str(self.phone_numbers.toPlainText()),
            "emails": str(self.emails.toPlainText()),
            "camera_name": str(self.camera_name.text()),
            "view_img": int(self.view_img.currentText())
            
        }
        

        flag = 1
        for i in self.cameras:
            if item["camera_name"] == i["camera_name"]:
                self.message_box.setText("This name exist please chooes another one")
                flag = 0
                
            
        if not check_connnections(item['source']):
            self.message_box.setText("The camera ip is not open")
            
        elif item['camera_name'] == '':
            self.message_box.setText("The camera name is empty")
        elif flag:
            self.cameras.append(item)
            self.list_of_cameras.clear()
            self.list_of_cameras.addItems([elem['camera_name'] for elem in self.cameras])
            self.message_box.setText(f"A new camera by the name {item['camera_name'] } added")
            
            self.camera_ip.clear()
            self.phone_numbers.clear()
            self.emails.clear()
            self.camera_name.clear()
            self.cordinate = [(0, 0), (0, 0)]  
            self.box_info.clear()
                       
        path = os.path.join('.', 'databases', f"{item['camera_name']}.db")
        if not os.path.exists(path):
            making_object_db(path)
            
    def clear_objects_db_service(self):
        cam_name = str(self.list_of_cameras.currentText())
        if os.path.exists(os.path.join('.', 'databases', f'{cam_name}.db')):
            os.remove(os.path.join('.', 'databases', f'{cam_name}.db'))
            making_object_db(os.path.join('.', 'databases', f'{cam_name}.db'))
    
    def RunCounter(self):
        for cam in self.cameras:
            print(cam)
            proc = multiprocessing.Process(target=detect, args=(cam,
                                                           ))
            proc.start()
            self.proc = proc
            self.process.append(proc)
        self.message_box.setText("App is runing")
         
    def StopCounter(self):
        for proc in self.process:
            if proc:
                proc.terminate()
        self.process = []
        self.message_box.setText("App is stoped")
            
    def TakeScreenShout(self):
        ip_cam = self.camera_ip.text()
        take_screen(str(ip_cam))
        
    def DrawLine(self):
        ip_cam = self.camera_ip.text()
        if not check_connnections(ip_cam):
            self.message_box.setText("The camera ip is not open")
        else:
            self.message_box.setText("please wait untill an screen shout happen")
            self.TakeScreenShout()
            self.message_box.setText("draw a line and press q")
            start, end = run_draw()
            self.cordinate = [start, end]
            self.line_info.setText(f"start:{str(start)} , end: {str(end)} ")
            self.message_box.setText("Line draw")

    def DrawBox(self):
        ip_cam = self.camera_ip.text()
        if not check_connnections(ip_cam):
            self.message_box.setText("The camera ip is not open")
        else:
            self.message_box.setText("please wait untill an screen shout happen")
            self.TakeScreenShout()
            self.message_box.setText("draw a box and press q")
            start, end = draw_boxes()
            self.cordinate = [start, end]
            self.box_info.setText(f"start:{str(start)} , end: {str(end)} ")
            self.message_box.setText("box draw")
         
        
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
