import cybw
from time import sleep
from env import Environment
from agent import Agent
from utilities import *

client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)

class AgentTrainer:
    def __init__(self, fileName='', maxIterate = 500, visualize = False, very_fast = True, algorithm = "Q_LEARNING",
                 epsilon0 = 0.9, epsilon_decrease = "EXPONENTIAL", epsilon_rate = -1):
        self.maxIterate = maxIterate
        self.fileName = fileName
        self.visualize = visualize
        self.very_fast = very_fast
        self.algorithm = algorithm
        self.agent = None
        self.epsilon0 = epsilon0
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_rate = epsilon_rate
        if(epsilon_rate == -1):
            self.epsilon_rate = 1 - 2/maxIterate

        assert algorithm in ["Q_LEARNING", "SARSA"]
        assert epsilon_decrease in ["LINEAR", "EXPONENTIAL"]

    def train(self):
        env = Environment()
        agent = None
        inited = False
        episode = 0

        while episode < self.maxIterate:
            while not Broodwar.isInGame():
                client.update()
                if not client.isConnected():
                    print("Reconnecting...")
                    reconnect()

            if (self.very_fast):
                Broodwar.setLocalSpeed(0)
                Broodwar.setGUI(False)

            if (not inited):
                inited = True
                Environment.initialize()
                agent = Agent([0, 1], Environment.state_size, self.fileName)

            Broodwar.sendText("black sheep wall")
            env.reset()

            last_state = -1
            last_action = -1
            step = 0

            while Broodwar.isInGame():
                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:

                        print("Episode %d ended in %d steps, epsilon : %.4f" % (episode+1, step, agent.epsilon))
                        print("Left enemy : %d, Score:, %d" % (len(Broodwar.enemy().getUnits()), env.getScore()))

                if (env.isActionFinished):
                    state = env.getCurrentState()
                    action = agent.getAction(state)
                    reward = env.getReward()
                    env.applyAction(action)

                    if (last_state >= 0):
                        if(self.algorithm is "Q_LEARNING"):
                            agent.learn_qlearning(last_state, last_action, state, reward)
                        elif(self.algorithm is "SARSA"):
                            agent.learn_sarsa(last_state, last_action, state, action, reward)

                    last_state = state
                    last_action = action
                    step += 1
                else:
                    env.doAction()
                if (self.visualize):
                    env.draw_circles()
                env.check_game_done()

                client.update()

            episode += 1
            if (self.epsilon_decrease is "LINEAR"):
                agent.epsilon = self.epsilon0 * (self.maxIterate - episode)/self.maxIterate
            elif(self.epsilon_decrease is "EXPONENTIAL"):
                agent.epsilon *= self.epsilonRate

        self.agent = agent
        exportTable(agent.q_table, self.algorithm, self.maxIterate)
