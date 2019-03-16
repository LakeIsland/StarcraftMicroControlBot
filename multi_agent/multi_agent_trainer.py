import cybw
import time
from deep_sarsa.deep_sarsa_env import *
import copy
import numpy as np
from multi_agent.state_extractor import *
import statistics
import pandas as pd
from simple_agent.epsilon_decay import *
import datetime
import random
from simple_agent.epsilon_decay import *
EPISODES = 1000
client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        time.sleep(0.5)


class MultiAgentTrainer:
    def __init__(self, socket, epsilon_decay:EpsilonDecay, very_fast=True, visualize=False, max_iterate=500, mode='train', file_or_folder_to_load=''
                 , algorithm = 'DeepSarsa', map_name = '', layers=[],
                 actor_layers=[], critic_layers=[],
                 export_per = -1, last_action_state_also_state=False, eligibility_trace = False):
        self.very_fast = very_fast
        self.socket = socket
        self.max_iterate = max_iterate
        self.visualize = visualize
        self.mode = mode
        assert mode in ['train', 'evaluate', 'evaluate_multiple_models']
        self.file_or_folder_to_load = file_or_folder_to_load
        if mode != 'train':
            assert file_or_folder_to_load != ''

        self.do_train = (mode == 'train')

        self.algorithm = algorithm
        assert algorithm in ['DeepSarsa','DQN','A2C']
        self.last_action_state_also_state = last_action_state_also_state
        self.state_size = 42
        self.action_size = 9
        self.map_name = map_name

        if self.last_action_state_also_state:
            self.nn_size = self.state_size * 2 + self.action_size
        else:
            self.nn_size = self.state_size

        self.export_per = export_per
        self.eligibility_trace = eligibility_trace

        self.socket.sendMessage(tag="init_info", msg={
            'max_iterate': max_iterate,
            'mode': mode,
            'file_or_folder_to_load': file_or_folder_to_load,
            'action_size':self.action_size,
            'state_size':self.nn_size,
            'map_name':map_name,
            'algorithm':algorithm,
            'layers':layers,
            'actor_layers':actor_layers,
            'critic_layers':critic_layers,
            'export_per':export_per,
            'eligibility_trace':eligibility_trace
        })

        tag, _ = self.socket.receiveMessage()
        assert tag == 'init finished'
        print(tag)

        self.total_iterate_count = -1
        self.total_iterate_counter = 0

        self.started_time = -1

        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        

    def set_epsilon_decay(self, epsilon_decay:EpsilonDecay):
        self.epsilon_decay = epsilon_decay

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_decay.get_epsilon(episode)

    def get_action(self, state, do_train=True):
        if (np.random.random() <= self.epsilon):
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

    def evaluate_multiple(self, test_file_path, test_zero=True, test_per = -1, test_iter = -1):
        result_file = open("../resultData/%s_test_result.txt" % test_file_path, 'w')

        self.socket.sendMessage(tag="test_multiple_model_info", msg ={
            'test_file_path': test_file_path,
            'test_per': test_per
        })

        tag, msg = self.socket.receiveMessage()
        assert tag == "file_number"
        print("%d files to test" % msg)
        self.total_iterate_count = test_iter * (msg + (1 if test_zero else 0))

        def eval_and_write_to_file(episode):
            result_info = self.evaluate(target_iterate=test_iter, close_socket=False, print_message=False)
            print("Win rate", result_info[0])
            print("Left avg", result_info[1])
            print("Score avg", result_info[3])

            result_file.write("%d\t" % episode)
            for i in result_info:
                result_file.write('%.4f\t' % i)
            result_file.write('\n')
            result_file.flush()

        if test_zero:
            eval_and_write_to_file(0)

        while True:
            self.socket.sendMessage(tag="load_file", msg=[11111])
            print("load file")
            tag, msg = self.socket.receiveMessage()
            print(tag)
            if tag == "load finished":
                print("%d trained model loaded:" % msg[0], msg[1])
                eval_and_write_to_file(msg[0])

            elif tag == "no more file to load":
                print("Test ended")
                break

            self.print_expected_time(time.time())

        self.socket.close()

    def evaluate(self, target_iterate = -1, close_socket = True, print_message = True, step_frame=5, max_frame=10000):
        episode = 0
        winEpisode = 0
        timeOutEpisode = 0
        score_list = []
        left_unit_list = []
        step_list = []

        if target_iterate == -1:
            target_iterate = self.max_iterate

        while episode < target_iterate:
            while not Broodwar.isInGame():
                client.update()
                if not client.isConnected():
                    print("Reconnecting...")
                    reconnect()

            if self.very_fast:
                Broodwar.setLocalSpeed(0)
                Broodwar.setGUI(False)

            #Broodwar.sendText("black sheep wall")

            last_states = {}
            last_actions = {}

            last_action_target = {}

            step = 0
            is_first = True
            last_frame_count = -1

            if self.started_time == -1:
                self.started_time = time.time()

            while Broodwar.isInGame():

                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:
                        if e.isWinner():
                            winEpisode += 1
                            step_list.append(step)

                        left_unit_list.append(len(Broodwar.self().getUnits()) - len(Broodwar.enemy().getUnits()))
                        score_list.append(get_score())
                        Broodwar.restartGame()

                    elif eventtype == cybw.EventType.MatchFrame:
                        if last_frame_count >=0 and Broodwar.getFrameCount() - last_frame_count < step_frame:
                            continue

                        if Broodwar.getFrameCount() > max_frame:
                            #print("Time over")
                            timeOutEpisode += 1
                            Broodwar.restartGame()
                            # Broodwar.leaveGame()

                        last_frame_count = Broodwar.getFrameCount()

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

                            action = self.get_action(nn_state, do_train=False)
                            target = apply_action(u, action)

                            last_actions[u.getID()] = action
                            last_action_target[u.getID()] = target

                        step += 1
                        is_first = False

                if self.visualize:
                    draw_action(last_actions, last_action_target)

                client.update()

            episode += 1
            self.total_iterate_counter += 1
            self.socket.sendMessage(tag="episode finished", msg=[episode])

            if print_message:
                #print("Left enemy : %d, Score: %d" % (len(Broodwar.enemy().getUnits()), get_score()))
                print("Win / Total : %d / %d, win rate : %.4f" % (winEpisode, episode, winEpisode / episode))
        if close_socket:
            self.socket.close()

        result_info = [0 for _ in range(6)]
        result_info[0] = winEpisode / episode
        result_info[1] = statistics.mean(left_unit_list)
        result_info[2] = statistics.stdev(left_unit_list)
        result_info[3] = statistics.mean(score_list)
        result_info[4] = statistics.stdev(score_list)
        result_info[5] = timeOutEpisode / episode
        result_info[6] = 0 if len(step_list) == 0 else statistics.mean(step_list)
        result_info[7] = 0 if len(step_list) == 0 else statistics.stdev(step_list)

        return result_info

    def print_expected_time(self, current_time):
        elapsed_time = current_time - self.started_time
        expected_left_time = elapsed_time * (self.total_iterate_count / self.total_iterate_counter) - elapsed_time
        print("Current iterate: %d/%d, Elapsed time: %s, Expected left time: %s" %
              (self.total_iterate_counter, self.total_iterate_count,
               str(datetime.timedelta(seconds=int(elapsed_time))),
               str(datetime.timedelta(seconds=int(expected_left_time))) ))

    def get_file_name(self, extend='txt'):
        now = datetime.datetime.now()
        nowDate = now.strftime('%Y_%m_%d_%H_%M')
        name = "../resultData/test_result_%s_%s_%s.%s"%(self.algorithm, self.map_name, nowDate,extend)
        return name

    def train(self, do_test_during_train=False, test_per=-1, test_iterate=-1, test_zero=True, step_frame=10, max_frame=10000):
        # env = DeepSARSAEnvironment()
        # agent = DeepSarsaAgent()
        if do_test_during_train:
            assert test_per != -1 and test_iterate != -1
            self.total_iterate_count = self.max_iterate + ((self.max_iterate // test_per) + (1 if test_zero else 0)) * test_iterate
        else:
            self.total_iterate_count = self.max_iterate

        episode = 0
        winEpisode = 0

        if do_test_during_train:
            f = open(self.get_file_name(), 'w')

        do_train = (self.mode == 'train')
        results = []
        self.started_time = time.time()

        def evaluate_and_write_to_file():
            result_info = self.evaluate(target_iterate=test_iterate, close_socket=False, print_message=False)
            print("Win rate", result_info[0])
            print("Left avg", result_info[1])
            print("Score avg", result_info[3])
            episode_result_info = [episode, self.epsilon] + result_info
            results.append(episode_result_info)

            for ind, i in enumerate(episode_result_info):
                if ind == 0:
                    f.write('%d\t' % (i))
                else:
                    f.write('%.4f\t' % (i))
            f.write('\n')
            f.flush()

        if do_test_during_train and test_zero:
            evaluate_and_write_to_file()

        while episode < self.max_iterate:
            while not Broodwar.isInGame():
                client.update()
                if not client.isConnected():
                    print("Reconnecting...")
                    reconnect()

            if (self.very_fast):
                Broodwar.setLocalSpeed(0)
                Broodwar.setGUI(False)

            # Broodwar.sendText("black sheep wall")

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

            self.epsilon = self.epsilon_decay.get_epsilon(episode)

            while Broodwar.isInGame():
                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:
                        if (do_train):
                            if self.algorithm == "A2C":
                                print("Episode %d ended in %d steps" % (episode + 1, step))
                            else:
                                print("Episode %d ended in %d steps, epsilon : %.4f" % (episode + 1, step, self.epsilon))
                            print("Left enemy : %d, Score: %d" % (len(Broodwar.enemy().getUnits()), get_score()))

                        if e.isWinner():
                            winEpisode += 1

                        Broodwar.restartGame()

                    elif eventtype == cybw.EventType.MatchFrame:
                        if last_frame_count >=0 and Broodwar.getFrameCount() - last_frame_count < step_frame:
                            continue

                        if Broodwar.getFrameCount() > max_frame:
                            print("Time over")
                            #Broodwar.leaveGame()
                            Broodwar.restartGame()

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

                            action = self.get_action(nn_state, do_train)
                            target = apply_action(u, action)

                            if (not is_first):
                                r_a = reward_attack(u, last_hit_points[u.getID()], last_cool_downs[u.getID()])
                                r_m = reward_move(u, last_states[u.getID()], last_actions[u.getID()], last_positions[u.getID()])

                                reward = r_a + r_m

                                if(do_train):
                                    last_nn_state = last_nn_states[u.getID()]
                                    last_action = last_actions[u.getID()]
                                    sarsa = [last_nn_state, last_action, reward, nn_state, action, u.getID(), 0]
                                    self.socket.sendMessage(tag="sarsa", msg=sarsa)
                                    tag, _ = self.socket.receiveMessage()
                                    assert tag == 'train finished'
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

                            sarsa = [last_nn_state, last_action, reward, last_nn_state, last_action, u.getID(), 1]
                            self.socket.sendMessage(tag="sarsa", msg=sarsa)
                            tag, _ = self.socket.receiveMessage()
                            assert tag == 'train finished'

                            last_destroyed_own_count += 1
                        else:
                            last_destroyed_enemy_count += 1

                if self.visualize:
                    draw_action(last_actions, last_action_target)

                client.update()

            episode += 1
            self.total_iterate_counter += 1

            if not do_train:
                print("Win / Total : %d / %d, win rate : %.4f" % (winEpisode, episode, winEpisode / episode))

            # if do_train:
            #     if self.epsilon_decrease == "LINEAR":
            #         self.epsilon = self.epsilon0 * (self.max_iterate - episode) / self.max_iterate
            #     elif self.epsilon_decrease == "EXPONENTIAL":
            #         self.epsilon *= self.epsilon_decay_rate
            #     elif self.epsilon_decrease == "INVERSE_SQRT":
            #         self.epsilon = self.epsilon0 / math.sqrt(1 + episode)

            if self.export_per != -1 and episode % self.export_per == 0:
                self.socket.sendMessage(tag="export", msg=[episode])
                tag, _ = self.socket.receiveMessage()
                assert tag == "export finished"

            self.socket.sendMessage(tag="episode finished", msg=[episode])

            if do_train and do_test_during_train and (episode % test_per == 0):
                evaluate_and_write_to_file()

            self.print_expected_time(time.time())

        if do_test_during_train:
            df = pd.DataFrame(results)
            df.columns = ['episode','epsilon','winrate','left_unit_avg','left_unit_stdev','score_avg', 'score_stdev']
            df.to_csv(self.get_file_name(extend='csv'), index=False)

        if self.export_per == -1:
            self.socket.sendMessage(tag="export", msg=[episode])
            tag, _ = self.socket.receiveMessage()
            assert tag == "export finished"

        self.socket.sendMessage(tag="end connection", msg=[11111])
        self.socket.close()