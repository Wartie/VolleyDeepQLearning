import random
import torch
import numpy as np

def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    Returns
    -------
    int
        ID of selected action
    """
    
    q_values = policy_net(state)
    return torch.argmax(q_values).item()
    
def select_reward_exploratory_action(state, policy_net, action_size, exploration, t, reward):
    random_val = random.random()
    if random_val < exploration.value(t, reward):
        return np.random.choice(action_size)
    else:
        return select_greedy_action(state, policy_net, action_size)

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """
    if random.random() < exploration.value(t):
        return np.random.choice(action_size)
    else:
        return select_greedy_action(state, policy_net, action_size)

def get_opposite_action(action_id):
    opposites = {0: 2,
                 1: 3,
                 2: 0,
                 3: 1}
    return opposites[action_id]

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    # gas, rot
    # return [[0, 0], [0.25, 0], [0.5, 0], [-0.25, 0], [0.1, 5], [0.1, -5], [-0.1, 5], [-0.1, -5], [0, 5], [0, -5]]
    # return [[1.5, 0], [0.5, 6], [0.5, -6], [0, 10], [0, -10]] #actual driving
    #return [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]] #8d
    
    return [[1, 0], [0, -1], [-1, 0], [0, 1]] #4d

    #return [[1.5, 0], [0, -8], [0, 8]]


    #return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [-0.5, 0.2, 0], [0.5, 0.2, 0], [0, 0.5, 0], [0, 0, 1.0], [0, 0, 0]]
