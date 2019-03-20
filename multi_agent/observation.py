import cybw
Broodwar = cybw.Broodwar
import math
from simulator.utilities import *
from multi_agent.state_extractor import apply_action
import numpy as np
from multi_agent.state_extractor import get_state_info

X_SIZE = 64
Y_SIZE = 64

X_PIXELS = X_SIZE * 8
Y_PIXELS = Y_SIZE * 8

UNIT_TYPE = 2
FORCE_TYPE = 3

STATE_SIZE = UNIT_TYPE + FORCE_TYPE + 1 + 1
MINIMAP_STATE_SIZE = FORCE_TYPE + 1 + 1

NON_SPATIAL_SIZE = 42
# class LocalObservationFrame:
#     def __init__(self, pixel_size=(X_PIXELS, Y_PIXELS), array_size=(X_SIZE,Y_SIZE)):
#         self.X_PIXELS, self.Y_PIXELS = pixel_size
#         self.X_SIZE, self.Y_SIZE = pixel_size


class Observation:
    def __init__(self, include_self):
        self.include_self=include_self

    def in_rectangle(self, o, v):
        from_left = v.x - o.x + X_PIXELS / 2
        from_top = v.y - o.y + Y_PIXELS / 2
        return from_left >= 0 and from_left <= X_PIXELS and from_top >=0 and from_top <=Y_PIXELS

    def to_observation_space(self, o, v):
        from_left = v.x - o.x + X_PIXELS / 2
        from_top = v.y - o.y + Y_PIXELS / 2

        lx = int(math.floor(from_left / X_PIXELS * X_SIZE))
        ly = int(math.floor(from_top / Y_PIXELS * Y_SIZE))

        lx = clamp(lx, 0, X_SIZE - 1)
        ly = clamp(ly, 0, Y_SIZE - 1)

        return lx, ly


    def to_minimap_space(self, v):
        lx = int(math.floor(v.x / Broodwar.mapWidth() * X_SIZE))
        ly = int(math.floor(v.y / Broodwar.mapHeight() * Y_SIZE))

        lx = clamp(lx, 0, X_SIZE - 1)
        ly = clamp(ly, 0, Y_SIZE - 1)

        return lx, ly

    def local_to_world_space(self, o, x, y):
        gx = (x / X_SIZE - 0.5) * X_PIXELS + o.x
        gy = (y / Y_SIZE - 0.5) * Y_PIXELS + o.y
        return gx, gy

    def minimap_to_world_space(self, x, y):
        gx = ((x+0.5) / X_SIZE) * Broodwar.mapWidth() * 32
        gy = ((y+0.5) / Y_SIZE) * Broodwar.mapHeight() * 32
        #print(gx, gy)
        return gx, gy

    def get_force(self, unit):
        if unit.getPlayer().getID() == Broodwar.self().getID():
            return 0
        elif unit.getPlayer().getID() == Broodwar.enemy().getID():
            return 1
        else:
            return 2

    def get_type(self, unit):
        if unit.getType() == cybw.UnitTypes.Protoss_Zealot:
            return 0
        elif unit.getType() == cybw.UnitTypes.Terran_Goliath:
            return 1
        # elif unit.getType() == cybw.UnitTypes.Zerg_Zergling:
        #     return 2
        # elif unit.getType() == cybw.UnitTypes.Zerg_Hydralisk:
        #     return 3
        #
        # return 3


    def get_health(self, unit):
        h = unit.getHitPoints() + unit.getShields()
        return h / 200.0


    def get_cooldown(self, unit):
        return unit.getGroundWeaponCooldown() / unit.getType().groundWeapon().damageCooldown()


    def get_minimap_state(self):
        total_map = self.get_minimap_state_np()
        s = total_map.tobytes()
        #print(total_map.shape)
        del total_map
        return s

    def get_minimap_state_np(self):
        local_force_map = np.zeros((X_SIZE, Y_SIZE, FORCE_TYPE), dtype=np.float32)
        local_terrain_map = np.zeros((X_SIZE, Y_SIZE, 1), dtype=np.float32)
        local_visible_map = np.zeros((X_SIZE, Y_SIZE, 1), dtype=np.float32)

        for ou in Broodwar.self().getUnits():
            if not ou.exists():
                continue
            oup = ou.getPosition()
            lx, ly = self.to_minimap_space(oup)
            local_force_map[lx, ly, self.get_force(ou)] = 1

        for ou in Broodwar.enemy().getUnits():
            if not ou.exists():
                continue
            oup = ou.getPosition()
            lx, ly = self.to_minimap_space(oup)
            local_force_map[lx, ly, self.get_force(ou)] = 1

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                gx, gy = self.minimap_to_world_space(x, y)
                wpos = cybw.WalkPosition(cybw.Position(gx, gy))
                tpos = cybw.TilePosition(cybw.Position(gx, gy))
                w = 1 if Broodwar.isWalkable(wpos) else 0
                v = 1 if Broodwar.isVisible(tpos) else 0
                local_terrain_map[x, y, 0] = w
                local_visible_map[x, y, 0] = v

        total_map = np.concatenate((local_force_map, local_terrain_map, local_visible_map), axis=2)
        #print(local_terrain_map.sum(), " TERRAIN SUM")
        del local_visible_map
        del local_terrain_map
        del local_force_map

        return total_map

    def get_non_spatial_state(self, own):
        #non_spatial_states = np.zeros((2, ))
        #typed_states = np.zeros((UNIT_TYPE, ))
        #typed_states[self.get_type(own)] = 1
        non_spatial_states = [0,0]
        non_spatial_states[0] = own.getGroundWeaponCooldown() / own.getType().groundWeapon().damageCooldown()
        non_spatial_states[1] = own.getHitPoints() / own.getType().maxHitPoints()

        #s = np.concatenate((typed_states, non_spatial_states)).tobytes()
        #del non_spatial_states

        return get_state_info(own)

        #return non_spatial_states

    def get_local_state(self, own):
        total_map = self.get_local_state_np(own)
        s = total_map.tobytes()
        del total_map
        return s

    def get_local_state_np(self, own):
        local_type_map = np.zeros((X_SIZE, Y_SIZE, UNIT_TYPE), dtype=np.float32)
        local_force_map = np.zeros((X_SIZE, Y_SIZE, FORCE_TYPE), dtype=np.float32)
        local_health_map = np.zeros((X_SIZE, Y_SIZE, 1), dtype=np.float32)
        local_terrain_map = np.zeros((X_SIZE, Y_SIZE, 1), dtype=np.float32)

        #local_type_map = np.zeros((X_SIZE, Y_SIZE))
        #local_type_map = np.zeros((X_SIZE, Y_SIZE))

        op = own.getPosition()
        for ou in Broodwar.self().getUnits():
            if not self.include_self and ou.getID() == own.getID():
                continue
            if not ou.exists():
                continue
            oup = ou.getPosition()
            if self.in_rectangle(op, oup):
                lx, ly = self.to_observation_space(op, oup)
                local_type_map[lx, ly, self.get_type(ou)] = 1
                local_force_map[lx, ly, self.get_force(ou)] = 1
                local_health_map[lx, ly, 0] = self.get_health(ou)

        for ou in Broodwar.enemy().getUnits():
            if not ou.exists():
                continue
            oup = ou.getPosition()
            if self.in_rectangle(op, oup):
                lx, ly = self.to_observation_space(op, oup)
                local_type_map[lx, ly, self.get_type(ou)] = 1
                local_force_map[lx, ly, self.get_force(ou)] = 1
                local_health_map[lx, ly, 0] = self.get_health(ou)

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                gx, gy = self.local_to_world_space(own.getPosition(), x, y)
                wpos = cybw.WalkPosition(cybw.Position(gx, gy))
                v = 1 if Broodwar.isWalkable(wpos) else 0
                local_terrain_map[x, y, 0] = v
        #print(local_terrain_map[:,:,0])
        total_map = np.concatenate((local_type_map, local_force_map, local_health_map, local_terrain_map), axis =2)

        del local_type_map
        del local_force_map
        del local_health_map
        del local_terrain_map
        return total_map

    def draw_minimap(self):
        u = self.get_minimap_state_np()
        w = 128
        h = 128
        Broodwar.drawBoxScreen(cybw.Position(0,0),
                               cybw.Position(w,h),
                               cybw.Colors.Red)
        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                if u[x][y][3] > 0.5:
                    Broodwar.drawBoxScreen(cybw.Position(x / X_SIZE * w, y / Y_SIZE * h),
                                        cybw.Position((x + 1) / X_SIZE * w, (y + 1) / Y_SIZE * h),
                                        cybw.Colors.Blue, True)
                if u[x][y][4] > 0.5:
                    Broodwar.drawBoxScreen(cybw.Position(x / X_SIZE * w, y / Y_SIZE * h),
                                           cybw.Position((x + 1) / X_SIZE * w, (y + 1) / Y_SIZE * h),
                                           cybw.Colors.Red, True)

    def draw_box(self):
        for u in Broodwar.self().getUnits():
            Broodwar.drawBoxMap(u.getPosition() - cybw.Position(X_PIXELS/2,Y_PIXELS/2),
                                u.getPosition() + cybw.Position(X_PIXELS/2,Y_PIXELS/2),
                                cybw.Colors.Red)

            a = self.get_local_state_np(u)

            for x in range(X_SIZE):
                for y in range(Y_SIZE):
                    # if a[x][y][0] == 1:
                    #
                    #     left_top = u.getPosition() - cybw.Position(X_PIXELS/2, Y_PIXELS/2)
                    #     Broodwar.drawBoxMap(left_top + cybw.Position(x / X_SIZE * X_PIXELS, y / Y_SIZE * Y_PIXELS),
                    #                         left_top + cybw.Position((x+1) / X_SIZE * X_PIXELS, (y+1) / Y_SIZE * Y_PIXELS),
                    #                         cybw.Colors.Red)
                    # el
                    if a[x][y][5] > 0.1:
                        left_top = u.getPosition() - cybw.Position(X_PIXELS / 2, Y_PIXELS / 2)
                        Broodwar.drawBoxMap(left_top + cybw.Position(x / X_SIZE * X_PIXELS, y / Y_SIZE * Y_PIXELS),
                                            left_top + cybw.Position((x + 1) / X_SIZE * X_PIXELS,
                                                                     (y + 1) / Y_SIZE * Y_PIXELS),
                                            cybw.Colors.Blue, True)
            del a
            break


