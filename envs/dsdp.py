import random
import numpy as np
from gym import spaces

class DiscreteStochasticDecisionProcess:
    def __init__(self):
        # left or right
        self.action_space = spaces.Discrete(2)
        # 6 possible states, from 0 to 5
        self.observation_space = spaces.Box(np.array([0]), np.array([5]), dtype=int)
        
        self.current_state = 1 # start in state 2
        self.visited_last = False
    
    # return format: current state, reward, is done, additional info
    def step(self, action):
        # going left
        if action == 0:
            self.current_state -= 1
        else:
            roll = random.random()
            if roll > 0.5 and self.current_state != 5:
                self.current_state += 1
            else:
                self.current_state -= 1
        
        if self.current_state == 5:
            self.visited_last = True
        if self.current_state == 0:
            return np.ndarray((1), buffer=np.array([self.current_state]), dtype=int), 1.0 / (1.0 if self.visited_last else 100.0), True, {}
        return np.ndarray((1), buffer=np.array([self.current_state]), dtype=int), 0.0, False, {}
    
    def reset(self):
        self.current_state = 1
        self.visited_last = False
        return np.ndarray((1), buffer=np.array([self.current_state]), dtype=int)
                    