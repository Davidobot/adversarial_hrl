import random
import numpy as np
from gym import spaces

TOTAL_STATES = 6

class DiscreteStochasticDecisionProcess:
    def __init__(self):
        # left or right
        self.action_space = spaces.Discrete(2)
        # 6 possible states, from 0 to 5
        self.observation_space = np.zeros(TOTAL_STATES)
        
        self.current_state = 1 # start in state 2
        self.visited_last = False
    
    def one_hot(self, n):
        buffer = np.zeros(TOTAL_STATES)
        buffer[n] = 1.0
        np.expand_dims(buffer, axis=0)
        # add batch dimension
        return np.ndarray((1, TOTAL_STATES), buffer=buffer, dtype=np.float)
    
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
            return self.one_hot(self.current_state), 1.0 / (1.0 if self.visited_last else 100.0), True, {}
        return self.one_hot(self.current_state), 0.0, False, {}
    
    def reset(self):
        self.current_state = 1
        self.visited_last = False
        return self.one_hot(self.current_state)
                    