import cybw
Broodwar = cybw.Broodwar
import math
from utilities import *

DIM_DIRECTION = 8
UNIT_THETA = 2 * math.pi / DIM_DIRECTION

def xy_direction_to_index(x, y):
    theta = math.atan2(y, x)
    theta = 2 * math.pi + theta + UNIT_THETA * 0.5
    theta = math.fmod(theta, 2*math.pi)
    index = int(theta / UNIT_THETA)
    return clamp(index, 0, DIM_DIRECTION - 1)

def direction_to_index(v):
    return xy_direction_to_index(v.x, v.y)

def index_to_direction(idx):
    theta = UNIT_THETA * idx
    return math.cos(theta), math.sin(theta)

def distance_normalized(d, d0):
    if d > d0:
        v = 0.05
    else:
        v = 1 - 0.95 * d / d0
    return v

def distance_terrain_normalized(d, d0):
    if d > d0:
        v = 0
    else:
        v = 1 - d / d0
    return v

def get_terrain_distance(unit, direction):
    dx, dy = index_to_direction(direction)
    sight_range = unit.getType().sightRange()
    d = 0
    while d < sight_range:
        wpos = cybw.WalkPosition(unit.getPosition() + cybw.Position(d * dx, d * dy))
        if not Broodwar.isWalkable(wpos):
            return d
        else:
            d += 20

    return d

def get_state_info(own):
    normalized_cooldown = own.getGroundWeaponCooldown() / own.getType().groundWeapon().damageCooldown()
    normalized_hitpoint = own.getHitPoints() / own.getType().maxHitPoints()

    own_sum_info = [0 for _ in range(DIM_DIRECTION)]
    own_max_info = [0 for _ in range(DIM_DIRECTION)]
    sight_range = own.getType().sightRange()

    for ou in Broodwar.self().getUnits():
        if ou.getID() == own.getID() or not ou.exists():
            continue
        idx = direction_to_index(ou.getPosition() - own.getPosition())
        nd = distance_normalized(ou.getDistance(own), sight_range)
        own_sum_info[idx] += nd
        own_max_info[idx] = max(nd, own_max_info[idx])

    enemy_sum_info = [0 for _ in range(DIM_DIRECTION)]
    enemy_max_info = [0 for _ in range(DIM_DIRECTION)]

    for eu in Broodwar.enemy().getUnits():
        if eu.getID() == own.getID() or not eu.exists():
            continue
        idx = direction_to_index(eu.getPosition() - own.getPosition())
        nd = distance_normalized(eu.getDistance(own), sight_range)
        enemy_sum_info[idx] += nd
        enemy_max_info[idx] = max(nd, enemy_max_info[idx])

    terrain_info = [0 for _ in range(DIM_DIRECTION)]
    for i in range(DIM_DIRECTION):
        terrain_info[i] = distance_terrain_normalized(get_terrain_distance(own, i), sight_range)

    state = [normalized_cooldown, normalized_hitpoint] + own_sum_info + \
            own_max_info + enemy_sum_info + enemy_max_info + terrain_info

    return state

def getWeakestWithinRange(unit):
    weakest = None
    for u in unit.getUnitsInRadius(unit.getType().groundWeapon().maxRange()):
        if u.getPlayer().getID() == Broodwar.self().getID():
            continue
        if not u.exists():
            continue
        if (weakest == None or weakest.getHitPoints() + weakest.getShields() > u.getHitPoints() + u.getShields()):
            weakest = u
        elif(weakest.getHitPoints()+ weakest.getShields() == u.getHitPoints()+ u.getShields() and
             unit.getDistance(weakest) > unit.getDistance(u)):
            weakest = u
    return weakest

def get_closest_unit(unit):
    closestEnemy = None
    for enemy in Broodwar.enemy().getUnits():
        if (closestEnemy == None or unit.getDistance(enemy) < unit.getDistance(closestEnemy)):
            closestEnemy = enemy
    return closestEnemy

def get_attack_enemy_unit(unit):
    target = getWeakestWithinRange(unit)
    # if target is None:
    #     target = get_closest_unit(unit)
    return target

def apply_action(unit, action):
    if(action == DIM_DIRECTION):
        target = get_attack_enemy_unit(unit)
        if target is not None:
            unit.stop()
            unit.attack(target)
        else:
            unit.holdPosition()

    else:
        dx, dy = index_to_direction(action)
        target = unit.getPosition() + cybw.Position(dx * 64, dy * 64)
        unit.rightClick(target)

    return target

lastAgentCount = 0

def reward_attack(unit, last_hitpoint, last_cool_down):
    r = 0
    if last_cool_down < unit.getGroundWeaponCooldown():
        r += unit.getType().groundWeapon().damageAmount() * unit.getType().groundWeapon().damageFactor()
    if unit.getHitPoints() + unit.getShields() < last_hitpoint:
        r -= (160 * 4) / (125 * 3) * (last_hitpoint - (unit.getHitPoints() + unit.getShields()))
    return r * 0.1

def reward_destroy():
    global lastAgentCount
    agentCount = len(Broodwar.self().getUnits())
    reward = -10 * (lastAgentCount - agentCount)
    lastAgentCount = agentCount
    return reward

def reward_move(unit, last_state, last_action, last_position):
    if len(Broodwar.enemy().getUnits()) == 0:
        ours = 0
        for u in unit.getUnitsInRadius(unit.getType().sightRange()):
            if u.getPlayer().getID() == Broodwar.self().getID():
                ours += 1
        if ours > 0:
            return 0
        last_own_max_info = last_state[10:10+DIM_DIRECTION]
        if last_action == DIM_DIRECTION or last_own_max_info[last_action] < 0.01:
            return -5
        else:
            return 0

    else:
        enemies = 0
        reward = 0
        for u in unit.getUnitsInRadius(unit.getType().sightRange()):
            if u.getPlayer().getID() != Broodwar.self().getID():
                enemies += 1

        if enemies == 0:
            last_enemy_max_info = last_state[26:26 + DIM_DIRECTION]
            if last_action == DIM_DIRECTION or last_enemy_max_info[last_action] < 0.01:
                reward -= 0.5

        # cannot move
        if last_action < DIM_DIRECTION:
            if unit.getPosition() == last_position:
                reward -= 0.5

            # own_unit = last_state[10:18]
            # ene_unit = last_state[26:34]
            # terrains = last_state[34:42]
            # if terrains[last_action] > 0.7 or own_unit[last_action] > 0.99 or ene_unit[last_action] > 0.9:
            #     reward -= 2
                #print("Last action was", last_action)
                #print("Terrain:", terrains[last_action], "Own:", own_unit[last_action], "Ene:", ene_unit[last_action])
                #print("CANNOT MOVE")

        return reward

    return 0

def get_score():
    global lastEnemyHealthSum, lastAgentHealthSum
    enemyHealthSum = 0
    agentHealthSum = 0
    for u in Broodwar.getAllUnits():
        if u.getPlayer().getID() is not Broodwar.self().getID():
            enemyHealthSum += (u.getHitPoints()+u.getShields())
        else:
            agentHealthSum += (u.getHitPoints()+u.getShields())

    score = agentHealthSum - enemyHealthSum
    return score


def draw_action(last_actions, last_action_targets):
    for u in Broodwar.self().getUnits():
        if not u.exists():
            continue
        if last_action_targets.get(u.getID(),None) is None:
            continue
        if last_actions[u.getID()] == DIM_DIRECTION:
            Broodwar.drawLineMap(u.getPosition(), last_action_targets[u.getID()].getPosition(), cybw.Colors.Red)
        else:
            Broodwar.drawLineMap(u.getPosition(), last_action_targets[u.getID()], cybw.Colors.White)
