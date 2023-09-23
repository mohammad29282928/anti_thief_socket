import socket
import json
import threading
import logging
import os
import cv2
import jdatetime

from utils_functions import send_call, send_sms, send_email


class Send_to_database_socket(threading.Thread):
    def __init__(self, obj):
        threading.Thread.__init__(self)
        self.obj      = {'key':"SendToServer", "Data":obj}
 
        # helper function to execute the threads
    def run(self):
        try:
            HOST = "127.0.0.1"  # The server's hostname or IP address
            PORT = 9000  # The port used by the server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))

                data = json.dumps(self.obj)
                s.sendall(data.encode())
                s.close()
        except:
            pass


def check_point_in_object(polygon, points):
    for point in points:
        if not polygon.contains(point):
            return False
    return True


def call_alarms(info):
    camera_name = info.get('camera_name')
    obj_name = info.get('obj_name')
    objects_alarms = info.get('objects_alarms')
    phone_numbers = info.get('phone_numbers')
    email_addresses = info.get('email_addresses')

    text = f""" 
                    دوربین
                    {camera_name}
                    آبجکت
                    {obj_name}
                    را شناسایی کرد
                """
    if objects_alarms[obj_name]['sms']:
        for num in phone_numbers:
            try:
                send_sms(num, text)
                logging.info(f'for camera {camera_name} sms sent')
            except:
                logging.info(f'for camera {camera_name} sms could not sent')

    if objects_alarms[obj_name]['call']:
        for num in phone_numbers:
            try:
                send_call(num, "مشکلی در دوربین ها رخ داده است")
                logging.info(f'for camera {camera_name} call sent')
            except:
                logging.info(f'for camera {camera_name} call could not sent')

    if objects_alarms[obj_name]['email']:
        for em in email_addresses:
            try:
                send_email(em, text)
                logging.info(f'for camera {camera_name} email sent')
            except:
                logging.info(f'for camera {camera_name} email could not sent')

        

    if objects_alarms[obj_name]['alarm']:
        try:
            os.system('start alarm.mp3')
            logging.info(f'for camera {camera_name} alarm ringed')
        except:
            logging.info(f'for camera {camera_name} alarm could not ringed')
     

def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)
    return img
#..............................................................................

def try_or_convert_time(date_time):
    try:
        return jdatetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
    except:
        return jdatetime.datetime.now()



