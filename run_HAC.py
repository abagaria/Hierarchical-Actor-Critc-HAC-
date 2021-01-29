"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary
import pdb
import numpy as np

NUM_TEST_GOALS = 10
NUM_TRAINING_EPISODES = 1000
NUM_TESTING_EPISODES = 50

def run_HAC(FLAGS,env,agent, seed):

    # Print task summary
    print_summary(FLAGS,env)

    training_successes = train_loop(FLAGS, env, agent, seed)

    agent.save_model(NUM_TRAINING_EPISODES)
    print(f"[HAC-Train] Finished training. Mean success rate: {np.mean(training_successes)}")

    log = {}
    start_states = [env.mdp.sample_random_state() for _ in range(NUM_TEST_GOALS)]
    goal_states = [env.mdp.sample_random_state() for _ in range(NUM_TEST_GOALS)]

    for start, goal in zip(start_states, goal_states):
        testing_successes = test_loop(FLAGS, env, agent, start_state=start, goal_state=goal)
        log[f"{start}, {goal}"] = {"successes": testing_successes}

        with open(f"{env.name}_log_file.pkl", "wb+") as f:
            cpickle.dump(log, f)


def train_loop(FLAGS, env, agent, seed):

    FLAGS.test = False

    total_episodes = 0
    successes = []

    for episode in range(NUM_TRAINING_EPISODES):

        print("\nEpisode %d, Total Episodes: %d" % (episode, total_episodes))
        success = agent.train(env, episode, total_episodes)

        print(f"[Testing] Episode {episode} \t Success {success} \t Final State: {env.get_state()[:2]}")
        successes.append(success)

    return successes


def test_loop(FLAGS, env, agent, start_state, goal_state):

    FLAGS.test = True

    total_episodes = 0
    successes = []

    for episode in range(NUM_TESTING_EPISODES):

        reset(env, start_state, goal_state)
        success = agent.train(env, episode, total_episodes)
        print(f"[Testing] Episode {episode} \t Success {success}")
        successes.append(success)

    return successes

def reset(env, start, goal):
    env.reset_to_start_state(start)
    env.set_goal_state(goal)
