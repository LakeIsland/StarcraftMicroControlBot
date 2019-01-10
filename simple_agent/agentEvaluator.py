import cybw
from time import sleep
from simple_agent.env import Environment
from simple_agent.agent import Agent
client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)

class AgentEvaluator:
    def __init__(self, fileName, maxIterate = 100, visualize = False, very_fast = True):
        self.maxIterate = maxIterate
        self.fileName = fileName
        self.visualize = visualize
        self.very_fast = very_fast

    def evaluate(self):
        env = Environment()
        agent = None
        inited = False
        episode = 0
        winEpisode = 0
        while episode < self.maxIterate:
            while not Broodwar.isInGame():
                client.update()
                if not client.isConnected():
                    reconnect()

            if(self.very_fast):
                Broodwar.setLocalSpeed(0)
                Broodwar.setGUI(False)

            if (not inited):
                inited = True
                Environment.initialize()
                agent = Agent([0, 1], Environment.state_size, self.fileName)

            Broodwar.sendText("black sheep wall")
            env.reset()
            step = 0
            while Broodwar.isInGame():
                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:

                        if e.isWinner():
                            winEpisode += 1

                if(not env.done):
                    if (env.isActionFinished):
                        state = env.getCurrentState()
                        action = agent.getAction(state, JUST_FOR_TEST=True)
                        env.applyAction(action)
                        step += 1
                    else:
                        env.doAction()

                    if (self.visualize):
                        env.draw_circles()
                    env.check_game_done()

                client.update()

            episode += 1
            print("Win / Total : %d / %d, win rate : %.4f" % (winEpisode, episode, winEpisode / episode))