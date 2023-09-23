import os
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from sort import *

from datetime import datetime
import jdatetime 
from sms_ir import SmsIr
import smtplib, ssl
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import json
from tensorflow.keras.models import load_model

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import uuid
import json
from tracer_utils import Send_to_database_socket, check_point_in_object, call_alarms
from tracer_utils import compute_color_for_labels, draw_boxes, try_or_convert_time


def detect(params):
    source      = params.get('serial','0')
    view_img =     params.get('view_img', 0)


    details = params.get('details', [])
    email_addresses = [item.get('email') for item in details]
    phone_numbers = [item.get('phone') for item in details]
    print('phone_numbers', phone_numbers, 'email_addresses', email_addresses)
    camera_name = params.get('name','None')
    objects = params.get('objects', [])
    blurratio = 40

    multi_models = [
        {'weight': 'weights/yolov5s.pt',  'data': 'data/coco128.yaml', 'name': 'public'},
        # {'weight': 'weights/yolov5s-face.pt',  'data': 'data/yolov5s-face.yaml', 'name': 'face'},
        # {'weight': 'weights/fire_v5.pt',  'data': 'data/fire_smoke.yaml', 'name': 'fire_smoke'},

    ]
        

    objects_alarms  = {item['objectId']:item for item in objects}

    save_txt, save_crop, nosave, agnostic_nms, augment, visualize, update, exist_ok, half   =False, False, False, False, False, False, False, False, False
    dnn, blur_obj, color_box, save_img= False, False, False, False
    imgsz=(640, 640)
    conf_thres=0.80
    iou_thres=0.35  
    max_det=1000
    device='cpu'
    classes=None
    project='runs/detect'
    name='exp'


    face_model = load_model(os.path.join('models', 'face_model.h5'))
    with open(os.path.join('models', 'labels.json')) as f:
        face_labels = json.load(f)

    sort_max_age = 10 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    objects_tracers = {}
    object_polygon = {}
    object_polygon_points = {}
    schedules = {}
    object_names= {}
    obj_static = {}
    for obj in objects:
        tracer = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
        
        objects_tracers[obj['objectId']] = tracer
        object_names[obj['objectId']] = str(obj.get('objectCode')).lower()
        if obj.get('objectBoundingBox', 0):
            points = list(eval(obj.get('objectBoundingBox')))
            if len(points) > 2 and points[0] != points[-1]:
                points.append(points[0])

            object_polygon[obj['objectId']] = Polygon(points)
            object_polygon_points[obj['objectId']] = points
            obj_static[obj['objectId']] = False
        else:
            obj_static[obj['objectId']] = True
        schedules[obj['objectId']] = {"scheduleName": obj.get('scheduleName'),
                                       'startDate': try_or_convert_time(obj.get('persianStartDate')), 
                                       'endDate': try_or_convert_time(obj.get('persianEndDate'))}
        

    print(object_names, schedules)
    # source = "rtsp://admin:2928awat@192.168.1.156:554/avstream/channel=1/stream=0-mainstream.sdp"
    #......................... 
    
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  




    detection_models = []
    device = select_device(device)
    for item in multi_models:
        model  = DetectMultiBackend(item.get('weight'), device=device, dnn=dnn, data=item.get('data'))
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        imgsz = check_img_size(imgsz, s=stride)  

        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
        if pt or jit:
            model.model.half() if half else model.model.float()

        if item.get('name') == 'face':
            names = {0:'face', 1:'face', 2:'face'}
        detection_models.append({"model": model, "names": names, 'name': item.get('name')})

    frameRate = params.get('frameRate')
    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=frameRate)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=frameRate)
        bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    dt, seen = [0.0, 0.0, 0.0], 0



    

    last_alarm_ring_time = jdatetime.datetime(1399, 6, 14, 17, 21, 19, 44596)
    for path, im, im0s, vid_cap, s, success in dataset:

        object_in = 0
        object_out = 0
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
        # Inference
        for model in detection_models:
            # print("model name is ", model.get('name'))
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model['model'](im, augment=augment, visualize=visualize)
            # print(len(pred), t1)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {model['names'][int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if blur_obj:
                            crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                            blur = cv2.blur(crop_obj,(blurratio,blurratio))
                            im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                        else:
                            continue
                    #..................USE TRACK FUNCTION....................
                    #pass an empty array to sort

                    for obj_name, sort_tracker  in objects_tracers.items():
                        
                        # print("obj_name, ", obj_name)
                        object_in = 0
                        object_out = 0
                       
                        dets_to_sort = np.empty((0,6))
                        
                        # NOTE: We send in detected object class too
                        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                            
                            if model.get('name') == 'face':                           
                                xyxy = [x1,y1,x2,y2]
                                crop_file_path = os.path.join('media', 'croped', f'{str(uuid.uuid4())}.jpg')
                                save_one_box(xyxy, im0, file=Path(crop_file_path))
                                image_size = (180, 180)
                                img = keras.utils.load_img(crop_file_path, target_size=image_size)
                                # plt.imshow(img)

                                img_array = keras.utils.img_to_array(img)
                                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                                # print("image saved ")
                                predictions = face_model.predict(img_array, verbose=None)

                                label, score = face_labels[str(predictions.argmax(axis=-1)[0])], predictions.max()

                                # print(f"label is {label} and score is {score} and obj_name is {object_names[obj_name]}")
                                if score > 0.5:
                                    if obj_static[obj_name] and  label == object_names[obj_name]:
                                        object_in += 1
                                    elif label == object_names[obj_name]:
                                        dets_to_sort = np.vstack((dets_to_sort, 
                                                                            np.array([x1, y1, x2, y2, 
                                                                                        conf, detclass])))
                                    
                            elif obj_static[obj_name] and  model['names'][int(detclass)] == object_names[obj_name]:
                                # print("temp", obj_static[obj_name], model['names'][int(detclass)])
                                object_in += 1

                            else:
                                if model['names'][int(detclass)] == object_names[obj_name]:
                                                    dets_to_sort = np.vstack((dets_to_sort, 
                                                                            np.array([x1, y1, x2, y2, 
                                                                                        conf, detclass])))
                                
                        
                        # Run SORT
                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks =sort_tracker.getTrackers()
                        

                        #loop over tracks
                        if not obj_static[obj_name]:
                            for track in tracks:
                                if len(track.centroidarr) >= 2:
                                    p1,p2 = Point(int(track.boxes[-1][0]), int(track.boxes[-1][1])), Point(int(track.boxes[-1][2]), int(track.boxes[-1][3]))
                                    p3,p4 = Point(int(track.boxes[-2][0]), int(track.boxes[-2][1])), Point(int(track.boxes[-2][2]), int(track.boxes[-2][3]))
                                    
                                    if (check_point_in_object(object_polygon[obj_name], [p1,p2]) and 
                                            not check_point_in_object(object_polygon[obj_name], [p3,p4])):
                                        object_in += 1

                                    if (not check_point_in_object(object_polygon[obj_name], [p1,p2]) and 
                                            check_point_in_object(object_polygon[obj_name], [p3,p4])):
                                        object_out += 1


          

                            [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                                    (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 

                        if not obj_static[obj_name]:
                            [cv2.line(im0, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                                        (124, 252, 0), thickness=3) for p1,p2 in  zip(object_polygon_points[obj_name],
                                            object_polygon_points[obj_name][1:]) ] 

                        alarm_flag = 0
                        # print(object_in, object_out, obj_is_occured)
                        if object_in  or object_out:
                            alarm_flag = 1


                        if alarm_flag: 
                            info = {
                                "camera_name": camera_name,
                                'obj_name': obj_name,
                                'objects_alarms': objects_alarms,
                                "phone_numbers": phone_numbers, 
                                "email_addresses":email_addresses, 
                            }

                            OBJ = {
                                "cameraId": params.get('cameraId'),
                                "objectId": obj_name,
                                "dateTime": jdatetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "input": True if object_in else False,
                                "output": True if object_out else False
                                }
                            
                            Send_to_database_socket(OBJ).start()
                            elapsed_time = jdatetime.datetime.now() - last_alarm_ring_time
                            if elapsed_time.total_seconds() > 1800:
                                if schedules[obj_name]['scheduleName'] == 'ever':
                                    call_alarms(info)
                                    last_alarm_ring_time = jdatetime.datetime.now()
                                    # Send_to_database_socket(OBJ).start()

                                elif schedules[obj_name]['scheduleName'] == 'selected_date':
                                    if jdatetime.datetime.now() > start_time and jdatetime.datetime.now() < schedules[obj_name]['endDate']:
                                        call_alarms(info)
                                        last_alarm_ring_time = jdatetime.datetime.now()
                                        # Send_to_database_socket(OBJ).start()


                                elif schedules[obj_name]['scheduleName'] == 'daily':
                                    now_time = jdatetime.datetime.now()
                                    if (now_time.hour > schedules[obj_name]['startDate'].hour and now_time.second > schedules[obj_name]['startDate'].second and
                                        now_time.hour < schedules[obj_name]['startDate'].hour and now_time.second < schedules[obj_name]['endDate'].second):
                                        call_alarms(info)
                                        last_alarm_ring_time = jdatetime.datetime.now()
                                        # Send_to_database_socket(OBJ).start()

                        # draw boxes for visualization
                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            draw_boxes(im0, bbox_xyxy, identities, categories, model['names'],color_box)
                            
                # print(start_point, end_point)

                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1) 
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path: 
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  
                            if vid_cap: 
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        # print("Frame Processing!")
        if cv2.waitKey(1) == ord('q'):
            break
    print("Video Exported Success")

    if vid_cap:
        vid_cap.release()




if __name__ == "__main__":
    detect()
