import numpy as np
from environment import Environment
from simple_rl.tasks.ant_maze.AntMazeMDPClass import AntMazeMDP
import pdb

class AntMazeEnv(Environment):
    def __init__(self, seed, dense_reward, vary_init, render):
        self.mdp = AntMazeMDP(seed, vary_init=vary_init, dense_reward=dense_reward, render=render)

        self.sim = self.mdp.env.wrapped_env.sim
        self.num_frames_skip = 5

        self.action_dim = self.mdp.action_space_size()
        self.state_dim = self.mdp.state_space_size()
        self.action_bounds = 30. * np.ones(self.action_dim)
        self.action_offset = np.zeros((len(self.action_bounds)))
        self.subgoal_dim = 2
        self.subgoal_bounds = [[-1, 4], [-1, 4]]

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # The goal space specifies the space from which we will sample the goal state
        # during training and testing. This domain will have the same goal state every time
        self.goal_position = self.mdp.goal_position
        x_goal = self.mdp.goal_position[0]
        y_goal = self.mdp.goal_position[1]
    
        self.goal_space_train = [[x_goal, x_goal], [y_goal, y_goal]]
        self.goal_space_test = [[x_goal, x_goal], [y_goal, y_goal]]
        self.end_goal_dim = len(self.goal_space_test)

        # Goal tolerance along each state dimension
        self.end_goal_thresholds = np.array([0.6, 0.6])
        self.subgoal_thresholds = np.array([0.6, 0.6])

        # Projection functions take in a state and squash the dimensions that do not matter
        # in terms of reaching the MDP's goal
        self.project_state_to_end_goal = self.project_to_goal
        self.project_state_to_subgoal = self.project_to_goal

        self.max_actions = 2000

        self.cumulative_reward = 0.
        self.cumulative_duration = 0

        self.name = "ant_maze_vary_init_{}_dense_reward_{}".format(vary_init, dense_reward)
        Environment.__init__(self, self.name)

    def project_to_goal(self, sim, state):
        """ Function should return the dimensions of the state that are relevant towards achieving the goal"""
        x = self.mdp.cur_state.position[0]
        y = self.mdp.cur_state.position[1]

        return np.array([x, y])

    def get_state(self):
        return self.mdp.cur_state.features()

    def reset_sim(self, training_time):
        self.mdp.reset(training_time=training_time)
        self.goal_position = np.copy(self.mdp.goal_position)
        return self.get_state()

    def execute_action(self, action):
        assert action.max() <= 30. and action.min() >= -30.
        
        reward, next_state = self.mdp.execute_agent_action(action)
        
        self.cumulative_reward += reward
        self.cumulative_duration += 1
        
        return self.get_state()

    def display_end_goal(self, end_goal):
        pass

    def get_next_goal(self, test):
        return self.goal_position

    def display_subgoals(self, subgoals):
        pass
