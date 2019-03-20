from deep_sarsa.deep_sarsa_agent_trainer import DeepSARSAAgentTrainer
from socketUtils.socketClient import *
from deep_q_agent_trainer import DeepQAgentTrainer
from simple_agent.agentTrainer import AgentTrainer
from simple_agent.agentEvaluator import AgentEvaluator
import sys, os
from multi_agent.multi_agent_trainer import MultiAgentTrainer
from simple_agent.archon_test import *
import time, datetime
from simple_agent.epsilon_decay import *
from multi_agent.multi_agent_trainer_map import MultiAgentTrainerCNN

def trainSimpleOne():
    trainer = AgentTrainer(maxIterate=1000, epsilon_decrease="LINEAR", algorithm="Q_LEARNING")
    try:
        trainer.train()
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Unexpected error:", sys.exc_info()[0])
        trainer.export()

def trainSimpleOne2():
    trainer = ArchonTest(maxIterate=1000, epsilon_decrease="LINEAR", algorithm="Q_LEARNING", visualize=True, very_fast=False)
    trainer.train()

def testSimpleOne():
    tester = AgentEvaluator(maxIterate=100,fileName="../tableData/Q_LEARNING_1000times_2019_01_11_00_31.txt",
                            visualize=True,very_fast=False)
    tester.evaluate()

def trainDeepSarsa():
    s = socketClient()
    s.accessToServer()
    trainer = DeepSARSAAgentTrainer(s, very_fast=True, visualize = False, max_iterate=500,
                                    file_to_load = '../modeldata/_deep_sarsa_500_times_2019_01_12_12_29.h5',
                                    mode='train',epsilon_decrease='LINEAR')
    trainer.train()

def testDeepSarsa():
    s = socketClient()
    s.accessToServer()
    trainer = DeepSARSAAgentTrainer(s, very_fast=True, visualize = False,
                                    max_iterate=100, mode = 'evaluate',
                                    file_to_load = '../modeldata/_deep_sarsa_500_times_2019_01_12_13_17.h5',
                                    )
    trainer.train()

def trainDQN():
    s = socketClient()
    s.accessToServer()
    trainer = DeepQAgentTrainer(s, very_fast=True, visualize=False, maxIterate=500)
    trainer.train()

def train_multi_agent():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=InvSqrtDecay(1.0), algorithm='DeepSarsa', very_fast=True, visualize = False, max_iterate=5000,
                                #file_to_load='../modeldata/DeepSarsa3G5Z/deep_sarsa_3G_5Z_5000_times_2019_01_31_08_54.h5',
                                mode='train',map_name='3G_4Z',layers=[100,100],
                                export_per=1000,last_action_state_also_state=True)
    trainer.train()

def train_multi_agent2():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=InvSqrtDecay(1.0), algorithm='DeepSarsa', very_fast=True, visualize = False, max_iterate=5000,
                                mode='train',map_name='3G_6Z',layers=[100, 100],
                                export_per=500,last_action_state_also_state=False,
                                eligibility_trace=False)

    trainer.train(do_test_during_train=False)

def train_multi_agent_with_eligibility():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=InvSqrtDecay(1.0), algorithm='DeepSarsa', very_fast=True, visualize = False, max_iterate=5000,
                                mode='train',map_name='3G_6Z',layers=[100, 100],
                                export_per=500,last_action_state_also_state=False,
                                eligibility_trace=True)

    trainer.train(do_test_during_train=False)
    #trainer.train(do_test_during_train=True, test_per=10, test_iterate=6, test_zero=True)

def train_multi_agent_a2c():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=ConstantEpsilon(0.0), algorithm='A2C', very_fast=True, visualize = False,
                                max_iterate=3000,
                                mode='train',map_name='3G_6Z', actor_layers=[100, 100], critic_layers=[100, 100],
                                export_per=1500,last_action_state_also_state=False)

    trainer.train(do_test_during_train=False)

def train_multi_agent_a2c_eligibility():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=ConstantEpsilon(0.0), algorithm='A2C', very_fast=True, visualize = False,
                                max_iterate=50000,
                                mode='train',map_name='3G_6Z', actor_layers=[100, 100], critic_layers=[100, 100],
                                export_per=1000,last_action_state_also_state=False, eligibility_trace=True)

    trainer.train(do_test_during_train=False)

def train_multi_agent_cnn():
    s = socketClient()
    s.accessToServer()
    #LinearDecay(1, 0.05, 2000)
    trainer = MultiAgentTrainerCNN(s, epsilon_decay=LinearDecay(1.0, 0.02, 1000), algorithm='CNN',
                                very_fast=True, visualize = False,
                                max_iterate=2000,
                                mode='train',map_name='3G_6Z_water',
                                export_per=500,last_action_state_also_state=False, eligibility_trace=False)

    trainer.train(do_test_during_train=False)

def test_multi_agent_cnn():
    s = socketClient()
    s.accessToServer()
    #LinearDecay(1, 0.05, 2000)
    trainer = MultiAgentTrainerCNN(s, epsilon_decay=ConstantEpsilon(0.0), algorithm='CNN', very_fast=False, visualize = True,
                                   file_or_folder_to_load='../modeldata/CNN_3G_6Z_water_2019_03_17_21_42/CNN_3G_6Z_water_4000_times_2019_03_18_16_37.h5',
                                max_iterate=100,
                                mode='evaluate',map_name='3G_6Z', last_action_state_also_state=False, eligibility_trace=False)

    trainer.train()

def test_multi_agent():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=ConstantEpsilon(0.0), algorithm='DeepSarsa', very_fast=True, visualize = False, max_iterate=100,
                                file_or_folder_to_load='../modeldata/deep_sarsa_3G_6Z_2019_02_06_23_53/deep_sarsa_3G_6Z_4000_times_2019_02_07_08_54.h5',
                                mode='evaluate', map_name='3G_6Z', layers=[100,100]
                                , last_action_state_also_state = False)
    #trainer.evaluate(100)

    trainer.train()


def test_multi_agent_multiple_models():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, epsilon_decay=ConstantEpsilon(0.0), algorithm='DeepSarsa', very_fast=True, visualize=False, max_iterate=100,
                                file_or_folder_to_load='../modeldata/multi_test',
                                mode='evaluate_multiple_models', map_name='3G_6Z', layers=[100, 100]
                                , last_action_state_also_state=False)

    trainer.evaluate_multiple(test_file_path="deep_sarsa_3G_6Z_2019_02_06_23_53", test_zero=True, test_iter=100)


def train_multi_agent_dqn():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, algorithm='DQN', very_fast=True, visualize = False, max_iterate=5000,
                                #file_to_load='../modeldata/deep_q_3G_3Z_500_times_2019_01_26_23_12.h5',
                                mode='train',epsilon_decrease='LINEAR',map_name='3G_4Z',layers=[100,100],
                                export_per=250)
    trainer.train()

def test_multi_agent_dqn():
    s = socketClient()
    s.accessToServer()
    trainer = MultiAgentTrainer(s, algorithm='DQN', very_fast=False, visualize = True, max_iterate=500,
                                file_or_folder_to_load='../modeldata/deep_q_3G_3Z_5000_times_2019_01_27_08_00.h5',
                                mode='evaluate', epsilon_decrease='LINEAR', map_name='3G_3Z', layers=[100,100])
    trainer.train()

if __name__ == "__main__":
    # eval = AgentEvaluator(fileName = "../q_table_test.txt")
    # eval.evaluate()
    start_time = time.time()
    #test_multi_agent()
    #trainSimpleOne2()
    # test_multi_agent()

    train_multi_agent_cnn()
    #test_multi_agent_cnn()
    #train_multi_agent_with_eligibility()

    print(str(datetime.timedelta(seconds=time.time()-start_time)))
    #trainSimpleOne()
    #testSimpleOne()

    #trainDQN()
    #trainDeepSarsa()

    #testDeepSarsa()


# def decoding(a):
#     a = a
#     b = a.decode()
#     print(b)
#
# temp = "클라이언트에서 서버로보내는 메세지입니다"
# msg = bytearray(temp, 'utf-8')
# try:
#     s = socketUtils.socketUtils(socketUtils.AF_INET, socketUtils.SOCK_STREAM)
#     print("소켓 생성완료")
# except socketUtils.error as err:
#     print("에러 발생 원인 :  %s" % (err))
#
# port = 1234
#
# s.connect(('localhost', port))
# #decoding(s.recv(1024))
# s.send(msg)
# s.close()
# import time
# s = socketClient()
# s.accessToServer()
# time.sleep(1)
# s.sendMessage([2.123,4.331,1.322,2.11])
# time.sleep(1)
# s.sendMessage([55.1223,4.331,1.322,2.11])