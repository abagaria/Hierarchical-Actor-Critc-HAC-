import numpy as np
from environment import Environment
from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP
import pdb

class PointMazeEnv(Environment):
    def __init__(self, seed, render):
        self.mdp = PointMazeMDP(seed, dense_reward=False, render=render)

        self.sim = self.mdp.env.wrapped_env.sim
        self.num_frames_skip = 4

        self.action_dim = self.mdp.action_space_size()
        self.state_dim = self.mdp.state_space_size()
        self.action_bounds = np.array([1., 1.])
        self.action_offset = np.zeros((len(self.action_bounds)))
        self.subgoal_dim = 3
        self.subgoal_bounds = [[-2, 10], [-2, 10], [0, 1]]

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # The goal space specifies the space from which we will sample the goal state
        # during training and testing. This domain will have the same goal state every time
        x_goal = self.mdp.goal_position[0]
        y_goal = self.mdp.goal_position[1]
        key_goal = 1
        self.goal_space_train = [[x_goal, x_goal], [y_goal, y_goal], [key_goal, key_goal]]
        self.goal_space_test = [[x_goal, x_goal], [y_goal, y_goal], [key_goal, key_goal]]
        self.end_goal_dim = len(self.goal_space_test)

        # Goal tolerance along each state dimension
        self.end_goal_thresholds = np.array([0.6, 0.6, 0.])
        self.subgoal_thresholds = np.array([0.6, 0.6, 0.])

        # Projection functions take in a state and squash the dimensions that do not matter
        # in terms of reaching the MDP's goal
        self.project_state_to_end_goal = self.project_to_goal
        self.project_state_to_subgoal = self.project_to_goal

        self.max_actions = 5000

        self.debug_actions = []
        self.debug_rewards = []
        self.cumulative_reward = 0.

        self.name = "four-room"
        Environment.__init__(self, self.name)

    def project_to_goal(self, sim, state):
        """ Function should return the dimensions of the state that are relevant towards achieving the goal"""
        x = self.mdp.cur_state.position[0]
        y = self.mdp.cur_state.position[1]
        key = self.mdp.cur_state.has_key

        return np.array([x, y, key])

    def get_state(self):
        return self.mdp.cur_state.features()

    def reset_sim(self):
        self.mdp.reset()
        return self.get_state()

    def execute_action(self, action):
        reward, next_state = self.mdp.execute_agent_action(action)
        self.cumulative_reward += reward
        self.debug_actions.append(np.copy(action))
        self.debug_rewards.append(reward)
        return self.get_state()

    def display_end_goal(self, end_goal):
        pass

    def get_next_goal(self, test):
        key = np.array([1])
        return np.concatenate((self.mdp.goal_position, key))

    def display_subgoals(self, subgoals):
        pass
