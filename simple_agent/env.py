from simulator.utilities import *
import cybw
import numpy as np
Broodwar = cybw.Broodwar

FEATURE_COUNT = 4
FLEE_FRAME = 12
DISTANCE_INDEX_N = 4
HEALTH_INDEX_N = 4

class Environment:
    def __init__(self):
        self.singleAgent = None
        self.lastAgentHealthSum = 0
        self.lastEnemyHealthSum = 0

        self.margin = 256.0
        self.repulsePower = 2
        self.FLEE_COEFF = 3

        self.isActionFinished = True
        self.lastStartAttack = False
        self.current_action = 0
        self.current_state = 0
        self.flee_counter = 0
        self.fleePosition = None
        self.attackTarget = None
        self.done = False


    featureNum = None
    state_size = 0

    @staticmethod
    def initialize():
        featureNum = [0 for _ in range(FEATURE_COUNT)]
        featureNum[0] = 2
        featureNum[1] = DISTANCE_INDEX_N
        featureNum[2] = len(Broodwar.enemy().getUnits()) + 1
        featureNum[3] = HEALTH_INDEX_N
        Environment.featureNum = featureNum
        Environment.state_size = int(np.prod(featureNum))

    def reset(self):
        self.isActionFinished = True
        self.done = False
        for u in Broodwar.self().getUnits():
            self.singleAgent = u


    def getReward(self):
        enemyHealthSum = 0
        agentHealthSum = 0
        for u in Broodwar.getAllUnits():
            if u.getPlayer().getID() is not Broodwar.self().getID():
                enemyHealthSum += u.getHitPoints()
            else:
                agentHealthSum += u.getHitPoints()

        reward = self.lastEnemyHealthSum - enemyHealthSum - (self.lastAgentHealthSum - agentHealthSum)
        self.lastEnemyHealthSum = enemyHealthSum
        self.lastAgentHealthSum = agentHealthSum
        return reward

    def getScore(self):
        enemyHealthSum = 0
        agentHealthSum = 0
        for u in Broodwar.getAllUnits():
            if (u.getPlayer().getID() is not Broodwar.self().getID()):
                enemyHealthSum += u.getHitPoints()
            else:
                agentHealthSum += u.getHitPoints()
        return (agentHealthSum - enemyHealthSum)


    def getCurrentState(self):
        indices = [0 for _ in range(FEATURE_COUNT)]
        indices[0] = self.getCooldownIndex()
        indices[1] = self.getDistancetoClosestEnemyIndex()
        indices[2] = self.getEnemyCount()
        indices[3] = self.getHealthIndex()
        index = 0
        for i in range(FEATURE_COUNT):
            index = index * Environment.featureNum[i]
            index = index + indices[i]
        return int(index)

    def getCooldownIndex(self):
        cooldown = self.singleAgent.getGroundWeaponCooldown()
        return 0 if cooldown > 0 else 1

    def getDistancetoClosestEnemyIndex(self):
        closestUnit = self.getClosestUnit()
        distance = 0
        if (closestUnit is None):
            distance = 100000
        else:
            distance = self.singleAgent.getDistance(closestUnit)
        index = clamp(distance // 32, 0, DISTANCE_INDEX_N - 1)
        return index

    def getClosestUnit(self):
        closestEnemy = None
        for enemy in Broodwar.enemy().getUnits():
            if (closestEnemy == None or self.singleAgent.getDistance(enemy) < self.singleAgent.getDistance(closestEnemy)):
                closestEnemy = enemy
        return closestEnemy

    def getWeakestUnit(self):
        weakestEnemy = None
        for enemy in Broodwar.enemy().getUnits():
            if (weakestEnemy == None or enemy.getHitPoints() < weakestEnemy.getHitPoints()):
                weakestEnemy = enemy
            elif (weakestEnemy.getHitPoints() == enemy.getHitPoints() and
                  self.singleAgent.getDistance(weakestEnemy) > self.singleAgent.getDistance(enemy)):
                weakestEnemy = enemy
        return weakestEnemy

    def getWeakestWithinRange(self):
        weakest = None
        for enemy in self.singleAgent.getUnitsInRadius(self.singleAgent.getType().groundWeapon().maxRange()):
            if (weakest == None or weakest.getHitPoints() > enemy.getHitPoints()):
                weakest = enemy
            elif(weakest.getHitPoints() == enemy.getHitPoints() and
                 self.singleAgent.getDistance(weakest) > self.singleAgent.getDistance(enemy)):
                weakest = enemy
        return weakest

    def getEnemyCount(self):
        return len(self.singleAgent.getUnitsInRadius(self.singleAgent.getType().groundWeapon().maxRange()))

    def getHealthIndex(self):
        index = self.singleAgent.getHitPoints() * HEALTH_INDEX_N // 80
        return clamp(index, 0, HEALTH_INDEX_N - 1)

    def getFleePosition(self):
        avgx = 0
        avgy = 0
        sx = self.singleAgent.getPosition().getX()
        sy = self.singleAgent.getPosition().getY()
        count = 0

        for u in Broodwar.enemy().getUnits():
            p = u.getPosition()
            dist = self.singleAgent.getDistance(u) + 0.01
            if (dist < 32 * 8):
                avgx = avgx + (sx - p.getX()) / dist
                avgy = avgy + (sy - p.getY()) / dist
                count = count + 1

        if (count is 0):
            return self.singleAgent.getPosition()

        length = math.sqrt(avgx * avgx + avgy * avgy)
        if(length > 0):
            vecx = avgx / length
            vecy = avgy / length
        else:
            print("LENGTH IS ZERO 11")

        leftR = max(0, self.margin - self.singleAgent.getPosition().getX()) / self.margin
        rightR = max(0, self.margin - (Broodwar.mapWidth() * 32 - self.singleAgent.getPosition().getX())) / self.margin
        upR = max(0, self.margin - self.singleAgent.getPosition().getY()) / self.margin
        downR = max(0, self.margin - (Broodwar.mapHeight() * 32 - self.singleAgent.getPosition().getY())) / self.margin

        vecx = vecx + (leftR - rightR) * self.repulsePower
        vecy = vecy + (upR - downR) * self.repulsePower

        length = math.sqrt(vecx * vecx + vecy * vecy)
        if(length > 0):
            vecx = vecx / length
            vecy = vecy / length
        else:
            print("LENGTH IS ZERO 22")

        npx = self.singleAgent.getPosition().getX() + int(32 * vecx * self.FLEE_COEFF)
        npy = self.singleAgent.getPosition().getY() + int(32 * vecy * self.FLEE_COEFF)

        npx = clamp(npx, 0, Broodwar.mapWidth() * 32)
        npy = clamp(npy, 0, Broodwar.mapHeight() * 32)
        return cybw.Position(npx, npy)

    def applyAction(self, action):
        # if(self.done):
        #     return
        if(action is 0):
            weakestUnit = self.getWeakestWithinRange()
            if(weakestUnit is not None):
                self.singleAgent.attack(weakestUnit)
                self.lastStartAttack = False
                self.attackTarget = weakestUnit
            else:
                weakestUnit = self.getWeakestUnit()
                if(weakestUnit is not None):
                    self.singleAgent.attack(weakestUnit)
                    self.lastStartAttack = False
                    self.attackTarget = weakestUnit
                else:
                    print("Weakest Unit in Null")
                    return

        elif(action is 1):
            self.flee_counter = FLEE_FRAME

        self.current_action = action
        self.isActionFinished = False

    def doAction(self):

        assert self.current_action == 0 or self.current_action == 1

        if self.current_action is 0:
            if self.lastStartAttack:
                self.isActionFinished = True
            self.lastStartAttack = self.singleAgent.isStartingAttack()

        elif self.current_action is 1:
            self.fleePosition = self.getFleePosition()
            self.singleAgent.move(self.fleePosition)
            if self.flee_counter <= 0:
                self.isActionFinished = True
            else:
                self.flee_counter -= 1

    def check_game_done(self):
        self.done = (len(Broodwar.enemy().getUnits()) == 0) or (len(Broodwar.self().getUnits())==0)

    def draw_circles(self):
        Broodwar.drawCircleMap(self.singleAgent.getPosition(), self.singleAgent.getType().groundWeapon().maxRange(),
                               cybw.Colors.Red)

        if(self.current_action is 1):
            Broodwar.drawCircleMap(self.fleePosition, 16, cybw.Colors.Red)
        else:
            Broodwar.drawCircleMap(self.attackTarget.getPosition(), 16, cybw.Colors.Red)

