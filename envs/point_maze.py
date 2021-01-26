import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PointMazeEnv(gym.Env):
    """
    Description:
        Simple and fast-to-run implementation of PointMaze, based on AntMaze.

    Source:
        Inspired by https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments

    Observation:
        Type: Box(3)
        Num     Observation               Min                     Max
        0       Point X co-ordinate       -1.5                    3.5
        1       Point Y co-ordinate       -1.5                    3.5
        2       Point Orientation         0.0                     1.0 (equivalent to 2pi)
        3       Time                      0.0                     max_steps / 10. (default: 50.0)

    Actions:
        Type: Box(2)
        Num   Action    Min         Max       
        0     Move      -1./scale   1./scale
        1     Rotate    -pi/4       pi/4

    Reward:
        100 if reached goal square; -0.1 otherwise for every timestep

    Starting State:
        Starting state is [U(-0.1, 0.1), U(-0.1, 0.1), (U(-0.1, 0.1) % (2pi)) / (2pi)]
        
        Note: moving with orientation of 0 is going to move the point right.
        The orientation increases anti-clockwise, so if in a state [0, 0, 0.25] (an orientation of pi/2)
        and executing action [0.5, 0] (move forward) will result in a new state [0, 0.5, 0.25]

    Episode Termination:
        Reaching the goal square. Time limit is left up to the user; recommended is 500 for a scaling factor of 4.
        
        The environment (for 500 steps max; scale_factor=4) is considered "solved" if an agent achieves an average reward of 90 or more over the latest 100 episode.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, scaling_factor=4, max_steps=500):
        self.maze = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        
        # scaling factor effects movement
        self.SCALING_FACTOR = scaling_factor
        self.maze_width = len(self.maze[0])
        self.maze_height = len(self.maze)
        
        # x, y, orientation
        self.starting_point = np.array([1.5, 1.5, 0])
        self.state_normalisation_factor = np.array([1., 1., 2 * math.pi])
        self.max_dist = 1. / self.SCALING_FACTOR
        self.max_turn = math.pi / 4

        # x_lim, y_lim, theta_lim
        high = np.array([self.maze_width - self.starting_point[0],
                         self.maze_height - self.starting_point[1],
                         1.0, max_steps / 10.],
                        dtype=np.float32)
        low = np.array([-self.starting_point[0],
                        -self.starting_point[1],
                        0.0, 0.0],
                       dtype=np.float32)
        
        # {max movement distance, max turn angle} in a single turn
        action_high = np.array([self.max_dist, self.max_turn], dtype=np.float32)
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.episode_steps = 0
        self.steps_beyond_done = None
        
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.episode_steps += 1
        
        ds = action[0]
        dtheta = action[1]
        
        x, y, theta = self.state
        
        # update theta and keep normalised to [0, 2pi] range
        theta = (theta + dtheta) % (2 * math.pi)
        # update position
        x = x + math.cos(theta) * ds
        y = y + math.sin(theta) * ds
        
        wall_collision = self.is_colliding(x, y, 1)
        if not wall_collision:
            self.state[0] = x
            self.state[1] = y
            self.state[2] = theta

        done = self.is_colliding(self.state[0], self.state[1], 'r')

        reward = -0.1
        
        if done and self.steps_beyond_done is None:
            # solved the maze!
            reward += 100.0
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        return self.normalised_state(), reward, done, {}
    
    # 1 is a solid wall; 'r' is the end square
    def is_colliding(self, x, y, check_for = 1):
        x = math.floor(x)
        y = math.floor(y)
        
        if x >= 0 and x < self.maze_width and y >= 0 and y < self.maze_height:
            return self.maze[self.maze_height - 1 - y][x] == check_for
        
        return True
        
    # starting point is always (0, 0); normalise theta to [0, 1]
    def normalised_state(self):
        return np.concatenate([(self.state - self.starting_point) / self.state_normalisation_factor, [self.episode_steps / 10.]])
    
    def reset(self):
        self.state = np.array(self.starting_point + np.random.uniform(-0.1, 0.1, self.starting_point.shape), dtype=np.float32)
        self.state[2] = self.state[2] % (2 * math.pi)
        self.steps_beyond_done = None
        
        self.episode_steps = 0
        return self.normalised_state()

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = self.maze_width
        block_size = screen_width/world_width
        
        point_size = block_size / 5.

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            self.point_trans = rendering.Transform()
            self.point_rot_trans = rendering.Transform()
            
            for yy in range(self.maze_height):
                y = self.maze_height - 1 - yy
                for x in range(self.maze_width):
                    if self.maze[y][x] != 0:
                        block = rendering.FilledPolygon([(0, 0), (0, block_size), (block_size, block_size), (block_size, 0)])
                        block_trans = rendering.Transform(translation=(x * block_size, yy * block_size))
                        block.add_attr(block_trans)
                        
                        if self.maze[y][x] == 1:
                            block.set_color(0.2, 0.2, 0.2)
                        elif self.maze[y][x] == 'r':
                            block.set_color(0.2, 0.2, 0.8)
                        
                        self.viewer.add_geom(block)
            
            point = rendering.make_circle(point_size / 2.)
            point.add_attr(self.point_trans)
            point.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(point)
            
            orientir = rendering.Line((0, 0), (0, 2 * point_size))
            orientir.linewidth.stroke = 5
            orientir.add_attr(self.point_rot_trans)
            orientir.add_attr(self.point_trans)
            orientir.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(orientir)          

        if self.state is None:
            return None

        x, y, theta = self.state
        self.point_trans.set_translation(x * block_size, y * block_size)
        self.point_rot_trans.set_rotation(theta - math.pi / 2) # tweak

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None