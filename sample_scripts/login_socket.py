# start socket (login)

import socket
import json


params = {
    "action": 'login',

    'data': {
        "username": 'mahmoodi',
        "password": "123456"
    }

}



HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 9000  # The port used by the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    data = json.dumps(params)
    s.sendall(data.encode())
    data = s.recv(1024).decode()
    print(data)
    s.close()
