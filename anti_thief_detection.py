import os
import sys
import cv2
import time
import torch
from pathlib import Path

import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)

from utils.torch_utils import select_device, time_sync

#---------------Object Tracking---------------
import sqlite3
import jdatetime 
import time
from sms_ir import SmsIr
import smtplib, ssl
import logging
import threading
import requests

from utils_functions import send_call, send_sms, send_email






def check_points_in_object(start_point, end_point, target_point1, target_point2):
    count = 0
    if (target_point1[0] <= max(start_point[0], end_point[0]) and 
        target_point1[0] >= min(start_point[0], end_point[0]) and 
        target_point1[1] <= max(start_point[1], end_point[1]) and 
        target_point1[1] >= min(start_point[1], end_point[1]) ):
        count += 1
    else:
        return 0
    if (target_point2[0] <= max(start_point[0], end_point[0]) and 
        target_point2[0] >= min(start_point[0], end_point[0]) and 
        target_point2[1] <= max(start_point[1], end_point[1]) and 
        target_point2[1] >= min(start_point[1], end_point[1]) ):
        count += 1
    else:
        return 0
    
    if count == 2:
        return 1
    




def check_point_in_object(polygon, points):
    for point in points:
        if not polygon.contains(point):
            return False
    return True
    

    

def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return 0
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return 0
    
    return 1

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 0
    return 1


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
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
        

def detect(params={}):
    logging.basicConfig(filename='./detection.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    source      = str(params.get('ip','0'))
    # source = "rtsp://192.168.137.219:8080/h264_ulaw.sdp"

    email_addresses = params.get('email_addresses','')
    phone_numbers = params.get('phone_numbers','')
    camera_name = params.get('name','None')
    view_img =     params.get('view_img', 0)
    start_time = params.get('start_time', '')
    end_time = params.get('end_time', '')
    objects = params.get('objects', [])

    scheduling_type = params.get('scheduling_type', 'ever')

    multi_models = {
        'weights/yolov5s.pt' : 'data/coco128.yaml',
        'weights/fire_v5.pt': 'data/fire_smoke.yaml'
    }

    objects_counter = {item['object_type']:0 for item in objects}
    objects_alarms  = {item['object_type']:item for item in objects}
    
    imgsz=(200, 300)
    conf_thres=0.25
    iou_thres=0.35  
    max_det=1000
    device='cpu'
      
    save_txt, save_conf, save_crop, nosave, agnostic_nms, augment, visualize, update, exist_ok, half   =False, False, False, False, False, False, False, False, False, False
    dnn, blur_obj = False, False
    classes=None
    project='runs/detect'
    name='exp'  
    print('test ')
    # source = "rtsp://admin:2928awat@192.168.1.156:554/avstream/channel=1/stream=0-mainstream.sdp"

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    # set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  

    detection_models = []
    device = select_device(device)

    for weights, data in multi_models.items():
        model  = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  

        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
        if pt or jit:
            model.model.half() if half else model.model.float()

        detection_models.append({"model": model, "names": names})
        logging.info(f'for camera {camera_name} model {weights} used')


    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1 

    dt, seen = [0.0, 0.0, 0.0], 0

    last_alarm_ring_time = jdatetime.datetime(1399, 6, 14, 17, 21, 19, 44596)
    for path, im, im0s, vid_cap, s, success in dataset:
        if not success:
            break
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        obj_counter = 0
        for model in detection_models:
            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model['model'](im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            for i, det in enumerate(pred):
                seen += 1
                if webcam: 
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    #..................USE TRACK FUNCTION....................
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        obj_counter += 1

                        info = {
                             "camera_name": camera_name,
                             'obj_name': model['names'][int(detclass)],
                             'objects_alarms': objects_alarms,
                             "phone_numbers": phone_numbers, 
                             "email_addresses":email_addresses, 
                        }

                        if model['names'][int(detclass)] in list(objects_counter.keys()):
                            objects_counter[model['names'][int(detclass)]] += 1
                            
                            elapsed_time = jdatetime.datetime.now() - last_alarm_ring_time
                            if elapsed_time.total_seconds() > 1800:
                                if scheduling_type == 'ever':
                                    call_alarms(info)
                                    last_alarm_ring_time = jdatetime.datetime.now()

                                elif scheduling_type == 'selected_date':
                                    if jdatetime.datetime.now() > start_time and jdatetime.datetime.now() < end_time:
                                        call_alarms(info)
                                        last_alarm_ring_time = jdatetime.datetime.now()

                                elif scheduling_type == 'daily':
                                    now_time = jdatetime.datetime.now()
                                    if (now_time.hour > start_time.hour and now_time.second > start_time.second and
                                        now_time.hour < end_time.hour and now_time.second < end_time.second):
                                        call_alarms(info)
                                        last_alarm_ring_time = jdatetime.datetime.now()



        time.sleep(2)
        # logging.info(f'model with camera {camera_name} {obj_counter} detected')
        if view_img:
            cv2.imshow("temp", im0)
            cv2.waitKey(1) 

        if cv2.waitKey(1) == ord('q'):
            break
    
    logging.info(f'for camera {camera_name} process ended or terminated')
    print("Video Exported Success")
    text = f""" 
                    دوربین
                    {camera_name}
                    مشکل پیدا کرده است
                """
    for num in phone_numbers:
        try:
            send_sms(num, text)
        except:
            pass


    if update:
        strip_optimizer(weights)
    
    if vid_cap:
        vid_cap.release()




# if __name__ == "__main__":
#     detect()
