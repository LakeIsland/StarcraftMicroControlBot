import cybw
from time import sleep
from deep_sarsa.deep_sarsa_env import *
import copy
import numpy as np
from multi_agent.state_extractor import *

EPISODES = 1000
client = cybw.BWAPIClient
Broodwar = cybw.Broodwar


def reconnect():
    while not client.connect():
        sleep(0.5)


class MultiAgentTrainer:
    def __init__(self, socket, very_fast=True, visualize=False, max_iterate=500, mode='train', file_to_load=''
                 , algorithm = 'DeepSarsa', epsilon_decrease='EXPONENTIAL', epsilon_decay_rate=-1, map_name = '', layers=[],
                 export_per = -1, last_action_state_also_state = False):
        self.very_fast = very_fast
        self.socket = socket
        self.max_iterate = max_iterate
        self.visualize = visualize
        self.mode = mode
        assert mode in ['train', 'evaluate']
        self.do_train = (mode == 'train')

        self.algorithm = algorithm
        assert algorithm in ['DeepSarsa','DQN']
        self.last_action_state_also_state = last_action_state_also_state
        self.state_size = 42
        self.action_size = 9

        if self.last_action_state_also_state:
            self.nn_size = self.state_size * 2 + self.action_size
        else:
            self.nn_size = self.state_size

        self.socket.sendMessage(tag="init_info", msg={
            'max_iterate': max_iterate,
            'mode': mode,
            'file_to_load': file_to_load,
            'action_size':self.action_size,
            'state_size':self.nn_size,
            'map_name':map_name,
            'algorithm':algorithm,
            'layers':layers,
            'export_per':export_per
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
            #return random.randint(0, 8)
            if(np.random.random() > 0.5):
                action = random.randint(0, 7)
            else:
                action = 8
        else:
            self.socket.sendMessage(tag="state", msg=state)
            tag, action = self.socket.receiveMessage()
            assert tag == "action"
            action = action[0]
        return action

    def train(self):
        # env = DeepSARSAEnvironment()
        # agent = DeepSarsaAgent()

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

            last_states = {}
            last_actions = {}
            last_nn_states = {}

            last_action_target = {}
            last_cool_downs = {}
            last_hit_points = {}
            last_positions = {}
            last_destroyed_own_count = 0
            last_destroyed_enemy_count = 0

            step = 0
            is_first = True
            last_frame_count = -1
            while Broodwar.isInGame():

                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:
                        if (self.do_train):
                            print("Episode %d ended in %d steps, epsilon : %.4f" % (episode + 1, step, self.epsilon))
                            print("Left enemy : %d, Score: %d" % (len(Broodwar.enemy().getUnits()), get_score()))

                        if e.isWinner():
                            winEpisode += 1

                        Broodwar.restartGame()

                    elif eventtype == cybw.EventType.MatchFrame:
                        if last_frame_count >=0 and Broodwar.getFrameCount() - last_frame_count < 10:
                            continue
                        last_frame_count = Broodwar.getFrameCount()
                        #print('frame: ', last_frame_count)
                        #print('destroyed:', last_destroyed_enemy_count, last_destroyed_own_count)
                        r_d = last_destroyed_own_count * -10 + last_destroyed_enemy_count * 10
                        last_destroyed_own_count = 0
                        last_destroyed_enemy_count = 0

                        for u in Broodwar.self().getUnits():
                            if not u.exists():
                                continue

                            state = get_state_info(u)
                            if self.last_action_state_also_state:
                                last_action = [0 for _ in range(9)]
                                if is_first:
                                    last_state = state
                                else:
                                    last_state = last_states[u.getID()]
                                    last_action[last_actions[u.getID()]] = 1

                                nn_state = state + last_state + last_action
                            else:
                                nn_state = state

                            action = self.get_action(nn_state, self.do_train)
                            target = apply_action(u, action)

                            if (not is_first):
                                r_a = reward_attack(u, last_hit_points[u.getID()], last_cool_downs[u.getID()])
                                r_m = reward_move(u, last_states[u.getID()], last_actions[u.getID()], last_positions[u.getID()])

                                reward = r_a + r_m

                                if(self.do_train):
                                    last_nn_state = last_nn_states[u.getID()]
                                    last_action = last_actions[u.getID()]
                                    sarsa = [last_nn_state, last_action, reward, nn_state, action, 0]
                                    self.socket.sendMessage(tag="sarsa", msg=sarsa)
                                    tag, _ = self.socket.receiveMessage()
                                    assert tag == 'trainFinished'
                                # agent.train_model(last_state, last_action, reward, state, action)

                            last_states[u.getID()] = state
                            last_nn_states[u.getID()] = nn_state

                            last_actions[u.getID()] = action
                            last_action_target[u.getID()] = target
                            last_cool_downs[u.getID()] = u.getGroundWeaponCooldown()
                            last_hit_points[u.getID()] = u.getHitPoints() + u.getShields()
                            last_positions[u.getID()] = u.getPosition()

                        step += 1
                        is_first = False

                    elif eventtype == cybw.EventType.UnitDestroy:
                        u = e.getUnit()
                        if u.getPlayer().getID() == Broodwar.self().getID():
                            reward = -20
                            last_nn_state = last_nn_states[u.getID()]
                            last_action = last_actions[u.getID()]

                            sarsa = [last_nn_state, last_action, reward, last_nn_state, last_action, 1]
                            self.socket.sendMessage(tag="sarsa", msg=sarsa)
                            tag, _ = self.socket.receiveMessage()
                            assert tag == 'trainFinished'

                            last_destroyed_own_count += 1
                        else:
                            last_destroyed_enemy_count += 1

                if self.visualize:
                    draw_action(last_actions, last_action_target)

                client.update()

            self.socket.sendMessage(tag="finish", msg=[11111])

            episode += 1
            if not self.do_train:
                print("Win / Total : %d / %d, win rate : %.4f" % (winEpisode, episode, winEpisode / episode))

            if (self.epsilon_decrease == "LINEAR"):
                self.epsilon = self.epsilon0 * (self.max_iterate - episode) / self.max_iterate
            elif (self.epsilon_decrease == "EXPONENTIAL"):
                self.epsilon *= self.epsilon_decay_rate
            elif (self.epsilon_decrease == "INVERSE_SQRT"):
                self.epsilon = self.epsilon0 / math.sqrt(1 + episode)
        self.socket.close()

