
import socket
import json
import multiprocessing
from tracer import detect
import socket
from threading import Thread
import cv2
import pickle
import  struct
import requests


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
    print("[Info]: Listening for connections on {0}, port {1}".format(self.host,self.port))
    while True:
        print("Hello, i am listening to your actions") # Just debug for now
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
                data = data.decode()
                data_loaded = json.loads(data)

                if data_loaded['action'] == 'login':
                    self.action_handler_thread.login_to_system(data_loaded['data'])
                    client.sendall('camera added'.encode())

                elif data_loaded['action'] == 'get_stream':
                    Thread(target = self.action_handler_thread.send_camera_stream, args = (data_loaded['data'],)).start() 
                    client.sendall('stream will be sen'.encode())
                    
                elif data_loaded['action'] == 'add_camera':
                    self.action_handler_thread.add_camera_to_list_of_camera(data_loaded)
                    client.sendall('camera added'.encode())

                elif data_loaded['action'] == 'stop':
                    self.action_handler_thread.stop_runing_active_cameras(data_loaded)
                    client.sendall('runing cameras stoped'.encode())
                    
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
        self.USERNAME = ''
        self.PASSWORD = ''


    def run(self):    # change here
        while 1:
            if len(self.updated):
                # print(self.updated)
                action = self.updated.pop()
                if action == 'add_camera':
                    self.restart_detection()
                elif action == 'stop':
                    self.stop_detection()

                elif action == 'update':
                    self.restart_detection()

                elif action == 'start_app':
                    self.restart_detection()

                
    
    def login_to_system(self, param_data):
        self.USERNAME = param_data.get('username')
        self.PASSWORD = param_data.get('password')
        user_info = requests.post('http://62.3.41.41/Account/Login', json  = {"username": "mahmoodi","password": "123456"})
        user_info = user_info.json()
        x = requests.get('http://62.3.41.41/Camera/Get', headers={'Authorization': user_info['data']['token']})
        data = x.json()
        if data.get('isSuccess'):
            self.cameras = [{'cameraId': 3, 'name': 'cm55', 
                             'code': 'cam3', 'ip': '192.168.1.156', 'port': 554,
                               'serial': 'rtsp://admin:2928awat@192.168.1.156:554/avstream/channel=1/stream=0-mainstream.sdp', 
                               'description': '', 'isActive': True, 'objects': [
                                   
                                   
                                    {
                                        "cameraObjectId": 1,
                                        "objectName": "person",
                                        "objectCode": "string",
                                        "objectId": 12,
                                        "schedule": 0,
                                        "boundingBox": [(0, 0), (0, 200), (200, 200), (0, 200), (0, 0)],
                                        "scheduleName": "ever",
                                        "startDate": "2023-09-11T18:50:25.005Z",
                                        "persianStartDate": '1402-06-21 00:27:54',
                                        "endDate": '1402-06-21 00:27:54',
                                        "persianEndDate": '1402-06-21 00:27:54',
                                        "alarm": True,
                                        "sms": True,
                                        "email": True,
                                        "call": True,
                                        "releh": True,
                                        "static": True
                                    }

                               ], 'details': [{"phone":"0920702921"}]}]
            # self.cameras = data.get('data', [])
            self.updated.append('start_app')


    def restart_detection(self):
        print("rsetarted ")
        for proc in self.process:
            if proc:
                proc.terminate()
        self.process = []
        
        for cam in self.cameras:
            if cam['isActive']:
                proc = multiprocessing.Process(target=detect, args=(cam, self.USERNAME, self.PASSWORD) )
                proc.daemon = True
                proc.start()
                self.process.append(proc)

    def stop_detection(self):
        print('stoped', self.process)
        for proc in self.process:
            if proc:
                proc.terminate()
        self.process = []

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
                print("get out")
                cap.release()
                break

    def add_camera_to_list_of_camera(self, param):
        self.cameras.append(param['data'])
        self.updated.append(param['action'])

    def stop_runing_active_cameras(self, param):
        self.updated.append(param['action'])


if __name__ == "__main__":
    print("[Info]: Loading complete.")
    multiprocessing.freeze_support()

    action_handler_thread = ThreadedHandleActionsServer()
    action_handler_thread.start()

    listen_to_client_thread = ThreadedListenSocketServer(action_handler_thread)  # change here
    listen_to_client_thread.start()






# while 1:
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.bind((HOST, PORT))
#     s.listen(1)
#     conn, addr = s.accept()
#     data_loaded = {}
#     with conn:
#         data = conn.recv(2048).decode()
#         data_loaded = json.loads(data)
#         # s.shutdown(socket.SHUT_RDWR) 
#         s.close()
    
#     if data_loaded['action']== 'start':

#         add_camera_to_list_of_camera(data_loaded['data'])
#         run_detection()
        
#     elif data_loaded['action']== 'stop':
#         print('temp')
#         stop_detection()
#         # break


