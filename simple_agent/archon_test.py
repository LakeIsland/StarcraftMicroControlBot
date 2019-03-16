import cybw
from time import sleep
from simple_agent.env import Environment
from simple_agent.agent import Agent
from simulator.utilities import *
import inspect

client = cybw.BWAPIClient
Broodwar = cybw.Broodwar

def reconnect():
    while not client.connect():
        sleep(0.5)
def methods(cls):
    return [x for x, y in cls.__dict__.items()]

def getClosestUnit(unit):
    closestEnemy = None
    for enemy in Broodwar.enemy().getUnits():
        if enemy.getType().isBuilding():
            continue
        if (closestEnemy == None or unit.getDistance(enemy) < unit.getDistance(closestEnemy)):
            closestEnemy = enemy
    return closestEnemy

def drawBullets():
    bullets = Broodwar.getBullets()
    for bullet in bullets:
        # print(bullet.getType())
        p = bullet.getPosition()
        velocityX = bullet.getVelocityX()
        velocityY = bullet.getVelocityY()
        lineColor = cybw.Colors.Green
        textColor = cybw.Text.Green
        a = bullet.getTargetPosition()
        # if bullet.getPlayer() == Broodwar.self():
        #     lineColor = cybw.Colors.Green
        #     textColor = cybw.Text.Green
        Broodwar.drawLineMap(p, a, cybw.Colors.Green)
        #Broodwar.drawLineMap(p, p+cybw.Position(velocityX, velocityY), lineColor)
        Broodwar.drawTextMap(p, chr(textColor) + str(bullet.getType()))

class ArchonTest:
    def __init__(self, fileName='', maxIterate = 500, visualize = False, very_fast = True, algorithm = "Q_LEARNING",
                 epsilon0 = 0.9, epsilon_decrease = "EXPONENTIAL", epsilon_decay_rate = -1):
        self.maxIterate = maxIterate
        self.fileName = fileName
        self.visualize = visualize
        self.very_fast = very_fast
        self.algorithm = algorithm
        self.agent = None
        self.epsilon0 = epsilon0
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_decay_rate = epsilon_decay_rate
        if(epsilon_decay_rate == -1):
            self.epsilon_decay_rate = 1 - 2/maxIterate

        assert algorithm in ["Q_LEARNING", "SARSA"]
        assert epsilon_decrease in ["LINEAR", "EXPONENTIAL","INVERSE_SQRT"]

        #parser = cybw.Unit()
        from types import FunctionType

        print(methods(cybw.Unitset))
        print(inspect.getmembers(cybw.Unit, predicate=inspect.ismethod))
        #print(inspect.getmembers(parser, predicate=inspect.ismethod))

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
                self.agent = agent

            Broodwar.sendText("black sheep wall")
            env.reset()

            last_state = -1
            last_action = -1
            step = 0
            Broodwar.setCommandOptimizationLevel(4)
            while Broodwar.isInGame():
                events = Broodwar.getEvents()
                for e in events:
                    eventtype = e.getType()
                    if eventtype == cybw.EventType.MatchEnd:
                        print("Episode %d ended in %d steps, epsilon : %.4f" % (episode+1, step, agent.epsilon))
                        print("Left enemy : %d, Score:, %d" % (len(Broodwar.enemy().getUnits()), env.getScore()))
                    elif eventtype == cybw.EventType.MatchFrame:
                        #Broodwar.self().getUnits().issueCommand(cybw.UnitCommandType)
                        #print(type(Broodwar.self().getUnits()))
                        #print(methods(type(Broodwar.self().getUnits())) )
                            #.useTech(cybw.TechTypes.Archon_Warp)
                        dod = False
                        for u in Broodwar.self().getUnits():
                            if not dod:
                                if u.getEnergy() < 75:
                                    continue
                                aa = getClosestUnit(u)
                                if aa is not None:
                                    u.useTech(cybw.TechTypes.Psionic_Storm, aa.getPosition())
                                    dod = True

                drawBullets()
                client.update()

            episode += 1
            if (self.epsilon_decrease == "LINEAR"):
                agent.epsilon = self.epsilon0 * (self.maxIterate - episode)/self.maxIterate
            elif(self.epsilon_decrease == "EXPONENTIAL"):
                agent.epsilon *= self.epsilon_decay_rate
            elif(self.epsilon_decrease == "INVERSE_SQRT"):
                agent.epsilon = self.epsilon0 / math.sqrt(1 + episode)

        exportTable(agent.q_table, self.algorithm, self.maxIterate)

    def export(self):
        exportTable(self.agent.q_table, self.algorithm, self.maxIterate)