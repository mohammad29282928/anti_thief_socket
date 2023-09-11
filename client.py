# echo-client.py

import socket
import json

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server


params = {
    "action": 'start',

    'data': {
        'email_addresses':[],
        'email_addresses': [],
        'name': 'cam1',
        'view_img': 0,
        'start_time':'',
        'end_time':'',
        'objects':[],
        'scheduling_type': 'ever',
        'active':True,
        'ip': '0'
    }

}


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    data = json.dumps(params)
    s.sendall(data.encode())
    s.close()
