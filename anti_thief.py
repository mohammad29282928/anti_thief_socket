from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType, loadUi
import sys
import sqlite3
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import pandas as pd
import jdatetime
import re
import os
import cv2 
import multiprocessing
from sms_ir import SmsIr
from anti_thief_database import making_object_db
from anti_thief_detection import detect
from utils_functions import initiat_data_base, send_sms, check_connnections
from utils_functions import internet_connection
from utils_functions import show_info_messagebox
from server_services import get_list_of_camera, convert_server_camera_to_system_camera
from server_services import remove_camera_from_server, edit_camera_from_server
from server_services import get_camera_update_from_server, del_camera_update_to_server
from server_services import send_objects_logs_to_server, login_to_system

from app_demo_utils import initiate_main_page_demo
from app_demo_utils import active_deactive_camera
from app_demo_utils import  make_camra_box

from add_camera import AddCamera
from edit_camera import EditCamera
from show_report import ShowReportCamera

ui, _ = loadUiType('main.ui')
class MainApp(QMainWindow,ui):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        initiate_main_page_demo(self)
        initiat_data_base() # removing all databases
        self.edited_camera = {}
        self.report_camera_selected = {}
        self.add_camera_button.clicked.connect(self.add_camera_button_service)
        self.go_to_login.clicked.connect(self.go_to_login_service)

    def go_to_login_service(self):
        self.login_button.setEnabled(True)
        self.program_main.setCurrentIndex(0)

    def add_camera_button_service(self):
        w = AddCamera(self)
        w.show()

    def show_list_of_camera(self):
        wrap = QWidget(self)
        v_layout = QVBoxLayout()
        for cam in self.cameras:
            v_layout.addWidget(make_camra_box(self, cam))
            v_layout.addStretch(3) 
        wrap.setLayout(v_layout)
        self.scrollArea.setWidget(wrap)

    def go_to_report_page(self):
        self.program_main.setCurrentIndex(1)
        self.show_list_of_camera()
        self.restart_detection_button_service()

    def send_logs_to_server(self):
        data = []
        try:
            with open('logs.txt', 'r') as file:
                lines = file.readlines()
                data = [[item.strip() for item in line.strip().split(',')] for line in lines]
                # file.write(f'{obj_name}, {str(count)}, {jdatetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") }\n')
        except:
            pass
        # print(data)
        if len(data):
            send_objects_logs_to_server(self.baseurl, {"owner": self.username, "data": data})
        try:
            with open('logs.txt', 'w') as file:
                pass
        except:
            pass

    def check_for_update(self):
        updates = get_camera_update_from_server(self.baseurl, self.username)


        for update in updates:
            del_camera_update_to_server(self.baseurl, update['id'], self.username)

        if len(updates):
            db_cams = get_list_of_camera(self.baseurl, self.username)
            update_cameras = []
            for cam in db_cams:
                temp = convert_server_camera_to_system_camera(cam)
                update_cameras.append(temp)

            for cam in self.cameras:
                    if cam['name'] not in [item['name'] for item in update_cameras]:
                        update_cameras.append(cam)
            
            self.cameras = update_cameras
                        
            print('update done')
            self.restart_detection_button_service()

    # report pages
    def restart_detection_button_service(self):
        self.stop_detection_service()
        self.run_detection_service()

    def remove_camera_button_service(self, selected_camera):
        select_camera = 0
        for cam in self.cameras:
            if cam['name'] == selected_camera['name']:
                select_camera = cam
        # print(select_camera)
        if select_camera:
            status = show_info_messagebox(f"آیا از حذف دوربین {select_camera['name']} مطمئن هستید؟ ")
            if status == 1024:
                out = []
                for cam in self.cameras:
                    if cam['name'] == select_camera['name']:
                        remove_camera_from_server(self.baseurl, cam, self.username)
                    else:
                        out.append(cam)
                self.cameras = out

                self.go_to_report_page()
        
                self.report_message.setText(f"دوربین {select_camera['name']}  حذف شد")   

    def active_deactive_button_service(self, selected_camera):
        active_deactive_camera(self, selected_camera)                   

    def edit_camera_button_service(self, selected_camera):
        print(selected_camera)
        flag = 0
        for cam in self.cameras:
            if cam['name'] == selected_camera['name']:
                self.edited_camera = cam
                flag = 1
        if flag and self.edited_camera:
            w = EditCamera(self)
            w.show()

    def show_camera_report_service(self, selected_camera):
        self.report_camera_selected  = selected_camera
        w = ShowReportCamera(self)
        w.show()

    def run_detection_service(self):
        # checking camera names databases
        for cam in self.cameras:
            db_path = os.path.join('.', 'databases', f"{cam['name']}.db")
            if not os.path.exists(db_path):
                making_object_db(db_path)
        # run app
        for cam in self.cameras:
            if cam['active']:
                try:
                    proc = multiprocessing.Process(target=detect, args=(cam,
                                                                ))
                    proc.start()
                    self.process.append(proc)
                except:
                    text = f"""
                        دوربین
                        {cam['name']} 
                        متوقف شده است لطفا پیگیری کنید
                    """
                    send_sms("09200702928", text)
        self.report_message.setText("برنامه در حال اجرا است")
        self.stop_detection.setEnabled(True)

    def stop_detection_service(self):
        self.stop_detection.setEnabled(False)
        for proc in self.process:
            if proc:
                proc.terminate()
        self.process = []
        self.report_message.setText("برنامه متوقف شد")

    # login pages
    def login_button_service(self):
        self.login_button.setEnabled(False)
        USERNAME = self.USERNAME.text()
        PASSWORD = self.PASSWORD.text()
        self.username = USERNAME
        self.password = PASSWORD
        

        if internet_connection():
            if login_to_system(self.baseurl, {'username': USERNAME, 'password': PASSWORD} ):
                db_cams = get_list_of_camera(self.baseurl, self.username)
                if len(db_cams):
                    for cam in db_cams:
                        # print(cam.get('name','None'), cam.get('cam_ip','None'))
                        if cam.get('name','None') in [cam['name'] for cam in self.cameras ]:
                            pass
                        elif not check_connnections(cam.get('cam_ip','None')):
                            pass
                        else:
                            # print(cam)
                            temp = convert_server_camera_to_system_camera(cam)
                            # print(temp)
                            self.cameras.append(temp)
    
                    if len(self.cameras):
                        self.go_to_report_page()
                    else:
                        self.program_main.setCurrentIndex(1)
                        w = AddCamera(self)
                        w.show()

                else:
                    self.program_main.setCurrentIndex(1)
                    w = AddCamera(self)
                    w.show()
            else:
                show_info_messagebox('نام کاربری  یا رمز عبور اشتباه است')
                self.login_button.setEnabled(True)
            # print(self.cameras)
        else:
            status = show_info_messagebox(' ارتباط شما برقرار نیست برای ادامه کلیک کنید ', False)
            if USERNAME == 'mohammad' and PASSWORD == '2928awat':
                    self.program_main.setCurrentIndex(1)
                    w = AddCamera(self)
                    w.show()
            else:
                show_info_messagebox('نام کاربری  یا رمز عبور اشتباه است')
                self.login_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        print(e)
        print("app faild")
        # text = """
        #             برنامه متوقف شده است لطفا پیگیری کنید
        #         """
        # send_sms("09200702928", text)