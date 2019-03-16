import socket
import pickle

class socketClient:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def accessToServer(self, port=1234):
        self.s.connect(('localhost', port))

    def sendMessage(self, tag, msg):
        packet = pickle.dumps([tag, msg])
        self.s.send(packet)
        #print("SEND",tag,msg)

    def receiveMessage(self):
        data = pickle.loads(self.s.recv(1024))
        tag = data[0]
        msg = data[1]
        #print(" ".join([str(x) for x in data]))
        #print("RECEIVE", tag, msg)
        return tag, msg

    # def sendMessage(self, msg):
    #     data = pickle.dumps(msg)
    #     self.s.send(data)
    #
    # def receiveMessage(self):
    #     data = pickle.loads(self.s.recv(1024))
    #     #print (" ".join([str(x) for x in data]))
    #     return data

    def close(self):
        self.s.close()