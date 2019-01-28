import cybw
from time import sleep
from deep_sarsa.deep_sarsa_env import *
import copy
import numpy as np

EPISODES = 1000
client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)

class DeepSARSAAgentTrainer:
    def __init__(self, socket, very_fast = True, visualize = False, max_iterate= 500, mode = 'train', file_to_load = ''
                 ,epsilon_decrease ='EXPONENTIAL', epsilon_decay_rate = -1):
        self.very_fast = very_fast
        self.socket = socket
        self.max_iterate = max_iterate
        self.visualize = visualize
        self.mode = mode
        assert mode in ['train', 'evaluate']
        self.do_train = (mode == 'train')

        self.socket.sendMessage(tag="init_info", msg={
            'max_iterate':max_iterate,
            'mode':mode,
            'file_to_load':file_to_load
            })

        self.epsilon0 = 1
        self.epsilon = self.epsilon0
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_decay_rate = epsilon_decay_rate
        if (epsilon_decay_rate == -1):
            self.epsilon_decay_rate = 1 - 2 / max_iterate

        assert epsilon_decrease in ["LINEAR", "EXPONENTIAL", "INVERSE_SQRT"]

    def get_action(self, state, do_train=True):
        if (np.random.random() <= self.epsilon and do_train):
            action = random.randint(0,1)
        else:
            self.socket.sendMessage(tag="state", msg=state)
            tag, action = self.socket.receiveMessage()
            assert tag == "action"
            action = action[0]
        return action

    def train(self):
        env = DeepSARSAEnvironment()
        #agent = DeepSarsaAgent()

        episode = 0
        winEpisode = 0

        while episode < self.max_iterate:
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
                        if(self.do_train):
                            print("Episode %d ended in %d steps, epsilon : %.4f" % (episode + 1, step, self.epsilon))
                            print("Left enemy : %d, Score:, %d" % (len(Broodwar.enemy().getUnits()), env.getScore()))

                        if e.isWinner():
                            winEpisode += 1

                if(not env.done):
                    if (env.isActionFinished):
                        state = env.getCurrentState()
                        # self.socket.sendMessage(tag="state", msg=state)
                        # tag, action = self.socket.receiveMessage()
                        # assert tag == "action"
                        # action = action[0]

                        #action = self.get_action(state,self.do_train)
                        #action = random.randint(0, 1)
                        # if (np.random.random() <= self.epsilon and self.do_train):
                        #     action = random.randint(0, 1)
                        # else:
                        #self.socket.sendMessage(tag="state", msg=state)
                        #tag, action = self.socket.receiveMessage()
                        #assert tag == "action"
                        #action = action[0]
                        action = self.get_action(state,self.do_train)

                        reward = env.getReward()
                        env.applyAction(action)

                        if (last_state is not None and self.do_train):
                            sarsa = [last_state, last_action,reward, state, action, 0]
                            self.socket.sendMessage(tag="sarsa", msg = sarsa)
                            tag, _ = self.socket.receiveMessage()
                            assert tag == 'trainFinished'
                            #agent.train_model(last_state, last_action, reward, state, action)

                        last_state = copy.deepcopy(state)
                        last_action = action

                        step += 1
                    else:
                        env.doAction()
                    if(self.visualize):
                        env.draw_circles()

                    env.check_game_done()
                    # if(env.done and len(Broodwar.self().getUnits())==0):
                    #     #print("This is End")
                    #     state = copy.deepcopy(last_state)
                    #     state[1] = 0
                    #     #state = env.getCurrentState()
                    #     #print('last',last_state)
                    #     #print('curr',state)
                    #     #self.socket.sendMessage(tag="state", msg=state)
                    #     #tag, action = self.socket.receiveMessage()
                    #     assert tag == "action"
                    #     #action = action[0]
                    #     action = last_action
                    #     reward = env.getReward()
                    #     self.socket.sendMessage(tag="sarsa", msg=[last_state, last_action, reward, state, action, 1])


                client.update()


            self.socket.sendMessage(tag="finish", msg=[11111])

            episode += 1
            if not self.do_train:
                print("Win / Total : %d / %d, win rate : %.4f" % (winEpisode, episode, winEpisode / episode))

            if (self.epsilon_decrease == "LINEAR"):
                self.epsilon = self.epsilon0 * (self.max_iterate - episode)/self.max_iterate
            elif(self.epsilon_decrease == "EXPONENTIAL"):
                self.epsilon *= self.epsilon_decay_rate
            elif(self.epsilon_decrease == "INVERSE_SQRT"):
                self.epsilon = self.epsilon0 / math.sqrt(1 + episode)
        self.socket.close()

