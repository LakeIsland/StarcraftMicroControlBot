import cybw
from time import sleep
from deep_sarsa.deep_sarsa_env import *
import copy

EPISODES = 1000
client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)

class DeepSARSAAgentTrainer:
    def __init__(self, socket, very_fast = True, visualize = False, maxIterate= 500):
        self.very_fast = very_fast
        self.socket = socket
        self.maxIterate = maxIterate
        self.visualize = visualize
        self.socket.sendMessage(tag="init_info", msg=[maxIterate])

    def train(self):
        env = DeepSARSAEnvironment()
        #agent = DeepSarsaAgent()

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

            Broodwar.sendText("black sheep wall")
            env.reset()

            last_state = None
            last_action = -1
            step = 0

            while Broodwar.isInGame():
                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:
                        #print("Episode %d ended in %d steps, epsilon : %.4f" % (episode + 1, step, agent.epsilon))
                        print("Left enemy : %d, Score:, %d" % (len(Broodwar.enemy().getUnits()), env.getScore()))

                if(not env.done):
                    if (env.isActionFinished):
                        state = env.getCurrentState()
                        self.socket.sendMessage(tag="state", msg=state)
                        #state = np.reshape(state,[1, agent.state_size])
                        tag, action = self.socket.receiveMessage()
                        assert tag == "action"
                        #print("action is ", " ".join([str(x) for x in action]))
                        action = action[0]

                        reward = env.getReward()
                        env.applyAction(action)

                        if (last_state is not None):
                            self.socket.sendMessage(tag="sarsa", msg = [last_state, last_action,
                                                                        reward, state, action, 0])
                           #agent.train_model(last_state, last_action, reward, state, action)

                        last_state = copy.deepcopy(state)
                        last_action = action

                        step += 1
                    else:
                        env.doAction()
                    if(self.visualize):
                        env.draw_circles()

                    env.check_game_done()
                    if(env.done):
                        state = env.getCurrentState()
                        self.socket.sendMessage(tag="state", msg=state)
                        tag, action = self.socket.receiveMessage()
                        assert tag == "action"
                        action = action[0]
                        reward = env.getReward()
                        self.socket.sendMessage(tag="sarsa", msg=[last_state, last_action, reward, state, action, 1])


                client.update()


            self.socket.sendMessage(tag="finish", msg=[11111])

            episode += 1

        #agent.model.save_weights("../save_model/deep_sarsa.h5")
        self.socket.close()

