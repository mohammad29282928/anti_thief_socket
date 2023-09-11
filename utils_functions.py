
import pandas as pd
import jdatetime
import re
import os
import cv2 

import glob
from sms_ir import SmsIr

import threading
import requests
from sms_ir import SmsIr
import smtplib, ssl

import socket
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import func_timeout




def internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False
    
def send_call_request(message, phone_number):
    api_key = "31354C6A7456485A3634347175414647526D436978416C7757757A4B6641355251583643796A76646B57733D"
    request = f"https://api.kavenegar.com/v1/{api_key}/call/maketts.json?receptor={phone_number}&&message={message}"
    r = requests.get(request)
    print(r.text)

def send_call(phone_number, message):
    p1 = threading.Thread(target=send_call_request, args=(message, phone_number, ))
    p1.start()
    p1.join()

def send_sms(phone, message):
    sms_ir = SmsIr(
    'R9R07HaxAugZy10YJ2U0iiC3um5YDNhef6XaeMyOHbU7o5db6AWB1iZ8wkHJE8Kz',
    '30007732901276',
    )

    sms_ir.send_sms(
    str(phone),
    str(message),
    '30007732901276',
    )

def send_email(email, message):

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "zaribar2928@gmail.com"  # Enter your address
    receiver_email = str(email)  # Enter receiver address
    password = "mptgqsixdjmwemph"
    message = str(message)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.encode("utf8") )

def check_opening_camera(cam_id):
    if str(cam_id).isdigit():
        cam_id = int(cam_id)
        cam1_ok = False 
        cam1 = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cam1.isOpened():
            cam1_ok = True
        return cam1_ok
    else:
        cam1_ok = False
        cam1 = cv2.VideoCapture(cam_id) 
        if cam1.isOpened():
            cam1_ok = True
        return cam1_ok

def run_timer_function(f, max_wait, default_value, arg):
    try:
        return func_timeout.func_timeout(max_wait, f, args=[arg])
    except func_timeout.FunctionTimedOut:
        pass
    return default_value
 
def check_connnections(cam_id, timer=5):
    x = run_timer_function(check_opening_camera, timer, False, cam_id)
    return x

def map_persian_name_to_english(persian_name):
    map_dict = {
        'آتش': 'fire', 
        'انسان': 'person',
        'کامیون': 'truck',
        'سواری': 'car',
        'دود': 'smoke'
    }
    return map_dict.get(persian_name, 0)

def map_english_name_to_persian(english_name):
    map_dict = {
        'آتش': 'fire', 
        'انسان': 'person',
        'کامیون': 'truck',
        'سواری': 'car',
        'دود': 'smoke'
    }
    new_dict = {v:k for k,v in map_dict.items()}
    return new_dict.get(english_name, 'انتخاب کنید')

def map_persian_scheduling_type_to_english(persian_name):
    map_dict = {
        'همیشه': 'ever',
        'روزانه':'daily',
        'دوره مشخص':'selected_date' 

    }
    return map_dict.get(persian_name, 0)

def map_english_scheduling_type_to_persian(english_name):
    map_dict = {
        'همیشه': 'ever',
        'روزانه':'daily',
        'دوره مشخص':'selected_date' 

    }
    new_dict = {v:k for k,v in map_dict.items()}
    return new_dict.get(english_name, 0)

def check_email_address(email):
 
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    if(re.fullmatch(regex, email)):
        return 1
 
    else:
        return 0

def initiat_data_base():
    files = glob.glob('databases/*')
    for f in files:
        os.remove(f)

    with open('logs.txt', 'w') as file:
        pass

def show_info_messagebox(message, flag=False):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
  
    # setting message for Message Box
    msg.setText(message)
      
    # setting Message box window title
    msg.setWindowTitle("پیام")
      
    # declaring buttons on Message Box
    if flag:
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    else:
        msg.setStandardButtons(QMessageBox.Ok)
      
    # start the app
    retval = msg.exec_()
    return retval

def get_idx_value_from_list(list_item, idx):
    try:
        if int(list_item[idx]):
            return True
        else:
            return False
    except:
        return False

def set_current_object_values(objects, idx, object_type, sms, email, call, alarm):
    out = {}
    try:
        out = objects[idx]
        # print(out)
        out['object_type'] = map_english_name_to_persian(out['object_type'])
        # print(out)
    except:
        out = {
            "object_type": 'انتخاب کنید',
            "sms": False,
            "email": False,
            "call": False, 
            "alarm": False}
    object_type.setCurrentText(out['object_type'])
    sms.setChecked(out['sms'])
    email.setChecked(out['email'])
    call.setChecked(out['call'])
    alarm.setChecked(out['alarm'])

def set_current_email_phone_values(emails, email1, email2):
    out = []
    for i in range(2):
        try:
            out.append(emails[i])

        except:
            out.append('')
    email1.setText(out[0])
    email2.setText(out[1])

