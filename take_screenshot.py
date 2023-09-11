
import cv2

def take_screen(camera=0):
    count = 0
    resize_shape = (640, 640)
#     cameras = "rtsp://admin:2928awat@192.168.1.156:554/avstream/channel=1/stream=0-mainstream.sdp"

    if str(camera).isdigit():
        camera = int(camera)
    while True:
        cam = camera
        vid = cv2.VideoCapture(cam)
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            count+=1
            if count < 3:
                continue
            else:
                count = 0
                vid = cv2.VideoCapture(cam)
                continue

        img = cv2.resize(img, resize_shape)       

        cv2.imwrite("screen.jpg", img)
        break
    
