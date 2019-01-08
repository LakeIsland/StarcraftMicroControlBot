import random
from env import Environment
from utilities import *

class Agent:
    def __init__(self, actions, state_number, fileName = '', learning_rate = 0.05, discount_factor = 0.9, epsilon0 = 0.9):
        if(fileName is ''):
            self.q_table = [[0] * len(actions) for _ in range(state_number)]
            assert len(self.q_table) is Environment.state_size
        else:
            self.q_table = importTable(fileName)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon0
        self.actions = actions

    def learn_qlearning(self, s, a, r, ns):
        current_q = self.q_table[s][a]
        next_state_qs = self.q_table[ns]
        max_q = max(self.q_table[ns])
        new_q = current_q + self.learning_rate * (r + self.discount_factor * max_q - current_q)
        self.q_table[s][a] = new_q

    def learn_sarsa(self, s, a, r, ns, na):
        current_q = self.q_table[s][a]
        next_state_qs = self.q_table[ns]
        n_q = self.q_table[ns][na]
        new_q = current_q + self.learning_rate * (r + self.discount_factor * n_q - current_q)
        self.q_table[s][a] = new_q

    def getAction(self, state, JUST_FOR_TEST=False):
        if (random.random() < self.epsilon and not JUST_FOR_TEST):
            action = random.randint(0,1)
        else:
            action = argmax(self.q_table[state])
        return action
