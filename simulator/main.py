from deep_sarsa.deep_sarsa_agent_trainer import DeepSARSAAgentTrainer
from socketUtils.socketClient import *
from deep_q_agent_trainer import DeepQAgentTrainer
from agentTrainer import AgentTrainer
from agentEvaluator import AgentEvaluator
import sys, os

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

if __name__ == "__main__":
    # eval = AgentEvaluator(fileName = "../q_table_test.txt")
    # eval.evaluate()

    #trainSimpleOne()
    #testSimpleOne()

    #trainDQN()
    #trainDeepSarsa()

    testDeepSarsa()

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