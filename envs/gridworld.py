import random
import numpy as np
from gym import spaces

# implementation of grid world environment as described in https://arxiv.org/abs/1810.10096
# The agent is initialized in an arbitrary location in an arbitrary room. The location of the key, the car, and the doorways are arbitrary and can vary

# world will be made out of ROOM_COUNT*ROOM_COUNT rooms with width/height equal to ROOM_SIZE
# key reward: +10
# key + car reward: +100
# bumping into wall: -2

# time limit for 5x5x2 is 200
# gamma = 0.99
# epsilon fixed at 0.2
# key reward +10
# lock reward +40

class GridWorld:
    def __init__(self, ROOM_SIZE=3, ROOM_COUNT=2):
        self.ROOM_SIZE = ROOM_SIZE
        self.ROOM_COUNT = ROOM_COUNT
        
        self.rand = np.random.default_rng()
        
        # N, S, W, E
        self.action_space = spaces.Discrete(4)
        
        self.reset()
        
        # player_pos, key_pos, car_pos + num_doors
        self.observation_space = np.zeros(3 + len(self.door_pairs()))
        
        self.num_positions = self.ROOM_SIZE * self.ROOM_SIZE * self.ROOM_COUNT * self.ROOM_COUNT
        
        
        self.MAX_STEPS = self.num_positions * 2
        self.steps = 0
        
        self.actions_to_move = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        
    
    def coord_to_pos(self, x, y):
        return x + self.ROOM_COUNT * self.ROOM_SIZE * y
    
    def pos_to_coord(self, pos):
        return pos % (self.ROOM_SIZE * self.ROOM_COUNT), pos // (self.ROOM_SIZE * self.ROOM_COUNT)
    
    def reset(self):
        self.player = self.coord_to_pos(*self.rand.integers(0, self.ROOM_SIZE * self.ROOM_COUNT, 2))
        self.key = self.coord_to_pos(*self.rand.integers(0, self.ROOM_SIZE * self.ROOM_COUNT, 2))
        self.car = self.coord_to_pos(*self.rand.integers(0, self.ROOM_SIZE * self.ROOM_COUNT, 2))
        
        # there are as many doors as there are rooms
        self.doors = {}
        self.door_pairing = []
        # grid lines, doors in format (door_pos, allowed_to_go_pos)
        for hor in range(self.ROOM_COUNT - 1):
            for i, x in enumerate(self.rand.integers(0, self.ROOM_SIZE, self.ROOM_COUNT)):
                x_coord = x + i * self.ROOM_SIZE
                y_coord = (hor + 1) * self.ROOM_SIZE
                
                if self.coord_to_pos(x_coord, y_coord) not in self.doors.keys():
                    self.doors[self.coord_to_pos(x_coord, y_coord)] = []
                if self.coord_to_pos(x_coord, y_coord - 1) not in self.doors.keys():
                    self.doors[self.coord_to_pos(x_coord, y_coord - 1)] = []
                self.doors[self.coord_to_pos(x_coord, y_coord)].append(self.coord_to_pos(x_coord, y_coord - 1))
                self.doors[self.coord_to_pos(x_coord, y_coord - 1)].append(self.coord_to_pos(x_coord, y_coord))
                
                self.door_pairing.append(self.coord_to_pos(x_coord, y_coord))
                self.door_pairing.append(self.coord_to_pos(x_coord, y_coord - 1))
                
        for vert in range(self.ROOM_COUNT - 1):
            for i, y in enumerate(self.rand.integers(0, self.ROOM_SIZE, self.ROOM_COUNT)):
                x_coord = (vert + 1) * self.ROOM_SIZE
                y_coord = y + i * self.ROOM_SIZE
                
                if self.coord_to_pos(x_coord, y_coord) not in self.doors.keys():
                    self.doors[self.coord_to_pos(x_coord, y_coord)] = []
                if self.coord_to_pos(x_coord - 1, y_coord) not in self.doors.keys():
                    self.doors[self.coord_to_pos(x_coord - 1, y_coord)] = []
                self.doors[self.coord_to_pos(x_coord, y_coord)].append(self.coord_to_pos(x_coord - 1, y_coord))
                self.doors[self.coord_to_pos(x_coord - 1, y_coord)].append(self.coord_to_pos(x_coord, y_coord))
                
                self.door_pairing.append(self.coord_to_pos(x_coord, y_coord))
                self.door_pairing.append(self.coord_to_pos(x_coord + 1, y_coord))
        
        self.visited_key = False
        self.steps = 0
        return np.concatenate(([self.player, self.key, self.car], self.door_pairs()))
    
    def door_pairs(self):
        return self.door_pairing
    
    def allowed(self, pos, new_pos):
        if pos in self.doors.keys():
            return new_pos in self.doors[pos]
        return False
    
    # return format: current state, reward, is done, additional info
    def step(self, action):
        self.steps += 1
        done = False
        reward = 0
        
        dx, dy = self.actions_to_move[action]
        px, py = self.pos_to_coord(self.player)
        new_pos = self.coord_to_pos(px + dx, py + dy)
        
        if px + dx < 0 or px + dx >= self.ROOM_SIZE * self.ROOM_COUNT or\
           py + dy < 0 or py + dy >= self.ROOM_SIZE * self.ROOM_COUNT:
            # can't leave rooms
            reward = -2
        else:
            # on border, need door to cross
            if (px % self.ROOM_SIZE == 0 and (px + dx) % self.ROOM_SIZE == self.ROOM_SIZE - 1) or\
               (px % self.ROOM_SIZE == self.ROOM_SIZE - 1 and (px + dx) % self.ROOM_SIZE == 0) or\
               (py % self.ROOM_SIZE == 0 and (py + dy) % self.ROOM_SIZE == self.ROOM_SIZE - 1) or\
               (py % self.ROOM_SIZE == self.ROOM_SIZE - 1 and (py + dy) % self.ROOM_SIZE == 0):
                if self.allowed(self.player, new_pos):
                    self.player = new_pos
                else:
                    reward = -2
            else:
                self.player = new_pos
        
        # check if picked up anything
        if self.player == self.key:
            self.key = -1
            self.visited_key = True
            reward = 10
        if self.player == self.car and self.visited_key:
            self.car = -1
            reward += 40
            done = True
        
        if self.steps > self.MAX_STEPS:
            done = True
        
        return np.concatenate(([self.player, self.key, self.car], self.door_pairs())), reward, done, {}
        
    def render(self):
        print("".join(['|', '_' * (self.ROOM_COUNT * self.ROOM_SIZE + self.ROOM_COUNT - 1), '|\n']), end='')
        for ry in range(self.ROOM_COUNT):
            for y in range(self.ROOM_SIZE):
                print("|", end='')
                for rx in range(self.ROOM_COUNT):
                    for x in range(self.ROOM_SIZE):
                        pos = self.coord_to_pos(x + rx * self.ROOM_SIZE, y + ry * self.ROOM_SIZE)
                        if pos == self.player:
                            print("P", end='')
                        elif pos == self.key:
                            print("K", end='')
                        elif pos == self.car:
                            print("C", end='')
                        elif pos in self.doors:
                            print("D", end='')
                        else:
                            print(".", end='')
                    print("|", end='')
                    #pos = self.coord_to_pos(self.ROOM_SIZE + rx * self.ROOM_SIZE, y + ry * self.ROOM_SIZE)
                    #if pos not in self.doors:
                    #    print("|", end='')
                    #else:
                    #    print(".", end='')
                print('\n', end='')
            print("|", end='')
            for rx in range(self.ROOM_COUNT):
                    for x in range(self.ROOM_SIZE):
                        print("_", end='')
                    print("|", end='')
            #for rx in range(self.ROOM_COUNT):
            #        for x in range(self.ROOM_SIZE):
            #            pos = self.coord_to_pos(x + rx * self.ROOM_SIZE, self.ROOM_SIZE + ry * self.ROOM_SIZE)
            #            if pos not in self.doors:
            #                print("_", end='')
            #            else:
            #                print(".", end='')
            #        print("|", end='')
            print('\n', end='')
            