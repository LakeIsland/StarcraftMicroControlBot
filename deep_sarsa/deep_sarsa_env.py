from simple_agent.env import Environment
from utilities import *

class DeepSARSAEnvironment(Environment):
    def getCurrentState(self):
        unit = self.singleAgent
        normalized_cooldown = unit.getGroundWeaponCooldown() / unit.getType().groundWeapon().damageCooldown()
        normalized_hitpoint = unit.getHitPoints() / unit.getType().maxHitPoints()
        normalized_enemy_counts = self.getEnemyCount() / 6
        normalized_enemy_closest_dist = self.getNormalizedDistancetoClosestEnemy()

        states = [normalized_cooldown, normalized_hitpoint, normalized_enemy_counts, normalized_enemy_closest_dist]

        return states

    def getNormalizedDistancetoClosestEnemy(self):
        closestUnit = self.getClosestUnit()
        distance = 0
        if (closestUnit is None):
            distance = 100000
        else:
            distance = self.singleAgent.getDistance(closestUnit)
        normalized_dist = clamp(distance / 48, 0, 1)
        return normalized_dist