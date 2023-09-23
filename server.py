
import socket
import json
import multiprocessing
from tracer import detect
from threading import Thread
import cv2
import pickle
import  struct
import requests
import logging
import threading

class Send_to_database_thread(threading.Thread):
    def __init__(self, token, obj):
        threading.Thread.__init__(self)
        self.token = token
        self.obj      = obj
 
        # helper function to execute the threads
    def run(self):
        try:
            # print("self.obj", self.obj)

            x = requests.post('http://62.3.41.41/ObjectDetection/Create', json =self.obj , headers={'Authorization': self.token})
        except Exception as e:
            print(e)



class ThreadedListenSocketServer(Thread):

  def __init__(self, action_handler_thread):
    Thread.__init__(self)  # change here
    self.host = "127.0.0.1"
    self.port = int(9000)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.bind((self.host, self.port))
    self.action_handler_thread = action_handler_thread

  def run(self):    # change here
    self.sock.listen(5)
    logging.info("[Info]: Listening for connections on {0}, port {1}".format(self.host,self.port))
    while True:
        logging.info("Hello, i am listening to your actions") # Just debug for now
        client, address = self.sock.accept()
        client.settimeout(60)
        Thread(target = self.listenToClient, args = (client,address)).start()   # change here

  def listenToClient(self, client, address):
    size = 1024
    while True:
        try:
            data = client.recv(size)
            if data:
                # Set the response to echo back the recieved data
                data_loaded = data.decode()
                
                data_loaded = json.loads(data_loaded)
                logging.info(f"data recived from server")
                
                data_loaded['Data'] = {k[0].lower() + k[1:]:v for k,v in data_loaded['Data'].items()}
                if data_loaded['Data'].get('objects', 0):
                    data_loaded['Data']['objects'] = [ {k[0].lower() + k[1:]:v for k,v in obj.items()} for obj in  data_loaded['Data']['objects']]

                if data_loaded['Data'].get('details', 0):
                    data_loaded['Data']['details'] = [ {k[0].lower() + k[1:]:v for k,v in obj.items()} for obj in  data_loaded['Data']['details']]

                if data_loaded['key'] == 'Token':
                    self.action_handler_thread.login_to_system(data_loaded['Data'])

                elif data_loaded['key'] == 'get_stream':
                    Thread(target = self.action_handler_thread.send_camera_stream, args = (data_loaded['Data'],)).start() 
                    
                elif data_loaded['key'] == 'CameraCreated':
                    self.action_handler_thread.camera_added(data_loaded['Data'])

                elif data_loaded['key'] == 'Stop':
                    self.action_handler_thread.stop_runing_active_cameras(data_loaded)

                elif data_loaded['key'] == 'CameraUpdated':
                    self.action_handler_thread.update_camera(data_loaded['Data'])

                elif data_loaded['key'] == 'CameraRemoved':
                    self.action_handler_thread.remove_camera(data_loaded['Data'])

                elif data_loaded['key'] == 'SendToServer':
                    self.action_handler_thread.send_data_to_server(data_loaded['Data'])
                    
            else:
                raise 'Client disconnected'
        except:
            client.close()
            return False



class ThreadedHandleActionsServer(Thread):

    def __init__(self):
        Thread.__init__(self)  # change here
        self.cameras = []
        self.updated = []
        self.process = []
        self.Token   = ''
    
    def login_to_system(self, param_data):
        self.Token = param_data.get('token')
        if not len(self.process):
            # print("self.Token", self.Token)
            x = requests.get('http://62.3.41.41/Camera/Get', headers={'Authorization':self.Token})
            data = x.json()
            if data.get('isSuccess'):
                self.cameras = data.get('data', [])
                self.restart_all_camera()

    def send_data_to_server(self, param_data):
        # logging.info(f"data {param_data}")
        Send_to_database_thread(self.Token, param_data).start()


    def camera_added(self, cam):
        logging.info(f"camera {cam.get('cameraId')} added  ")
        self.cameras.append(cam)
        if cam['isActive']:
            proc = multiprocessing.Process(target=detect, args=(cam, ) )
            proc.daemon = True
            proc.start()
            item = {'key': cam.get('cameraId'), 'proc': proc}
            self.process.append(item)


    def restart_one_camera(self, cam):

        selected_proess = 0
        for item in self.process:
            if cam.get('cameraId') == item['key']:
                selected_proess = item['proc']
                self.process.remove(item)
        if selected_proess:
            selected_proess.terminate()
        if selected_proess and cam['isActive']:
            proc = multiprocessing.Process(target=detect, args=(cam,) )
            proc.daemon = True
            proc.start()
            item = {'key': cam.get('cameraId'), 'proc': proc}
            self.process.append(item)


    def remove_one_camera(self, cam):

        selected_proess = 0
        for item in self.process:
            if cam.get('cameraId') == item['key']:
                selected_proess = item['proc']
                self.process.remove(item)
        if selected_proess:
            selected_proess.terminate()


    def restart_all_camera(self):
        logging.info("start runing pocess ")
        for item in self.process:
            if item['proc']:
                item['proc'].terminate()
        self.process = []
        
        for cam in self.cameras:
            if cam['isActive']:
                proc = multiprocessing.Process(target=detect, args=(cam, ) )
                proc.daemon = True
                proc.start()
                item = {'key': cam['cameraId'], 'proc': proc}
                self.process.append(item)


    def send_camera_stream(self, param_data):
        cap=cv2.VideoCapture(param_data.get('ip'))
        clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        HOST = '127.0.0.1'
        PORT = 7080

        clientsocket.connect((HOST,PORT))

        while True:
            ret,frame=cap.read()
            # Serialize frame
            data = pickle.dumps(frame)

            # Send message length first
            message_size = struct.pack("L", len(data)) ### CHANGED

            # Then data
            try:
                clientsocket.sendall(message_size + data)
            except:
                cap.release()
                break


    def update_camera(self, camera):
        logging.info(f"camera {camera.get('cameraId')} updated  ")
        out = []

        for cam in self.cameras:
            if cam.get('cameraId') == camera.get('cameraId'):
                out.append(camera)
            else:
                out.append(cam)
        self.cameras = out
        self.restart_one_camera(camera)


    def remove_camera(self, camera):
        logging.info(f"camera {camera.get('cameraId')} removed  ")
        out = []
        for cam in self.cameras:
            if cam.get('cameraId') == camera.get('cameraId'):
                pass
            else:
                out.append(cam)
        self.cameras = out
        self.remove_one_camera(camera)


    def stop_runing_active_cameras(self, param):
        logging.info('stoped', self.process)
        for proc in self.process:
            if proc:
                proc.terminate()
        self.process = []


if __name__ == "__main__":
    logging.info("[Info]: Loading complete.")
    multiprocessing.freeze_support()

    action_handler_thread = ThreadedHandleActionsServer()
    action_handler_thread.start()

    listen_to_client_thread = ThreadedListenSocketServer(action_handler_thread)  # change here
    listen_to_client_thread.start()





