import cybw
from time import sleep
from env import Environment
from agent import Agent

client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)

SHOW_BOX = True
JUST_FOR_TEST = True
fileName = "../q_table_test.txt"

env = Environment()
agent = None
inited = False
episode = 0

while True:
    print("waiting to enter match")
    while not Broodwar.isInGame():
        client.update()
        if not client.isConnected():
            print("Reconnecting...")
            reconnect()

    Broodwar.setLocalSpeed(0)

    if(not inited):
        inited = True
        Environment.initialize()
        agent = Agent([0, 1], Environment.state_size, fileName)

    Broodwar.sendText("black sheep wall")
    env.reset()

    last_state = -1
    last_action = -1
    step = 0
    alreadyDone = False
    while Broodwar.isInGame():
        events = Broodwar.getEvents()
        for e in events:
            eventtype = e.getType()
            if eventtype == cybw.EventType.MatchEnd:
                print("Episode %d ended in %d steps, epsilon : %.4f" % (episode, step, agent.epsilon))
                print("Left enemy : %d, Score:, %d" % (len(Broodwar.enemy().getUnits()), env.getScore()))
                if e.isWinner():
                    Broodwar << "I won the game\n"
                else:
                    Broodwar << "I lost the game\n"

        if(env.isActionFinished):
            state = env.getCurrentState()
            action = agent.getAction(state, JUST_FOR_TEST)
            reward = env.getReward()
            env.applyAction(action)

            if(last_state >=0 and not JUST_FOR_TEST):
                agent.learn_qlearning(last_state, last_action, state, reward)

            last_state = state
            last_action = action
            step += 1
        else:
            env.doAction()

        if(SHOW_BOX):
            env.draw_circles()
        env.check_game_done()


        client.update()
    episode += 1
