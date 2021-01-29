import numpy as np
from environment import Environment
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


class D4RLAntMazeEnv(Environment):
    def __init__(self, maze_type):

        self.mdp = D4RLAntMazeMDP(maze_size=maze_type)
        self.sim = self.mdp.env.sim
        self.num_frames_skip = 5

        self.action_dim = self.mdp.action_space_size()
        self.state_dim = self.mdp.state_space_size()
        self.action_bounds = np.ones(self.action_dim)
        self.subgoal_dim = 5

        # Taken from Andrew's code
        max_height = 1
        max_velo = 3

        x_high, y_high = self.mdp.get_x_y_high_lims()
        x_low, y_low = self.mdp.get_x_y_low_lims()
        self.subgoal_bounds = np.array([[x_low, x_high], [y_low, y_high],
                                        [0, max_height], [-max_velo, max_velo],
                                        [-max_velo, max_velo]])

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        self.goal_space_train = [[x_low, x_high], [y_low, y_high]]
        self.goal_space_test = [[x_low, x_high], [y_low, y_high]]
        self.end_goal_dim = len(self.goal_space_test)
        self.goal_position = self.mdp.sample_random_state()

        # Goal tolerance along each state dimension
        goal_dim = 2
        len_threshold = 0.6
        height_threshold = 0.2
        velo_threshold = 0.8
        self.end_goal_thresholds = len_threshold * np.ones((goal_dim,))
        self.subgoal_thresholds = np.array([len_threshold, len_threshold,
                                            height_threshold,
                                            velo_threshold, velo_threshold])

        # Projection functions take in a state and squash the dimensions that do not matter
        # in terms of reaching the MDP's goal
        self.project_state_to_end_goal = self.project_to_end_goal
        self.project_state_to_subgoal = self.project_to_subgoal

        self.max_actions = 1000
        self.cumulative_reward = 0.
        self.cumulative_duration = 0
        self.successes = []

        self.name = f"{self.mdp.env_name}-HAC"
        Environment.__init__(self, self.name)

    def project_to_end_goal(self, sim, state):
        return sim.data.qpos[:2]

    def project_to_subgoal(self, sim, state):
        return np.concatenate((sim.data.qpos[:3], sim.data.qvel[:2]))

    def get_state(self):
        return self.mdp.cur_state.features()

    def reset_sim(self, training_time):
        self.mdp.reset()
        if not training_time:
            s0 = self.mdp.get_position(self.mdp.sample_random_state())
            self.mdp.set_xy(s0)
        return self.get_state()

    def execute_action(self, action):
        _, next_state = self.mdp.execute_agent_action(action)

        reward, done = self.mdp.dense_gc_reward_function(next_state, self.goal_position, {})
        next_state.set_terminal(done)

        self.cumulative_reward += reward
        self.cumulative_duration += 1
        self.successes.append(done)

        return self.get_state()

    def display_end_goal(self, end_goal):
        pass

    def get_next_goal(self, test):
        return self.mdp.sample_random_state()

    def display_subgoals(self, subgoals):
        pass
