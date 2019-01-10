from socket.socketClient import *
import time
s = socketClient()
s.accessToServer()
for i in range(2):
    time.sleep(1)
    s.sendMessage([2.123,4.331,1.322,2.11])
    time.sleep(1)
    s.sendMessage([55.1223,4.331,1.322,2.11])