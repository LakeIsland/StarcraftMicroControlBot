from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
from utilities import *

class DeepSarsaAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = [0, 1]
        self.action_size = len(self.action_space)
        self.state_size = 4
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state, JUST_FOR_TEST=False):
        if (np.random.random() <= self.epsilon and not JUST_FOR_TEST):
            action = random.randint(0,1)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
        return action

    def train_model(self, state, action, reward, next_state, next_action):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        target = np.reshape(target,[1,2])
        self.model.fit(state, target, epochs=1, verbose=0)
