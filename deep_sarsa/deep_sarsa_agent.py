import numpy as np
from utilities import *

class DeepSarsaAgent:
    def __init__(self, socket):
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.socket = socket

    def get_action(self, state, JUST_FOR_TEST=False):
        if (np.random.random() <= self.epsilon and not JUST_FOR_TEST):
            action = random.randint(0,1)
        else:
            self.socket.sendMessage(tag="state", msg=state)
            tag, action = self.socket.receiveMessage()
            assert tag == "action"
            action = action[0]
        return action
