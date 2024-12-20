import numpy as np
import torch
import torch.optim as optim
import cv2 as cv
import copy
from action_set import get_action_set, select_exploratory_action, select_reward_exploratory_action, select_greedy_action, get_opposite_action
from learning import perform_qlearning_step, update_target_net, train_classification
from model import DQN, DQNLidar, DDQNLidar, BasicDQN, DQNSEQLidar, DQNLSTMLidar
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule, RewardBasedEGreedy
from utils import get_state, visualize_training
# initialize your carla env

def collect_data(env, save_root="C:/Users/wanga/OneDrive/Documents/BU/Coursework/LimoProject/data_seq_rel"):
    t = 0
    for i in range(150):
        obs, done = env.reset(i, False), False

        # Run each episode until episode has terminated or 600 time steps have been reached
        while not done:
            # action = actions[action_id]
            # print(action_id)
            done = env.user_input_step(t, save_root)
            t += 1


def evaluate(env, load_path='3en5_1m_8m_0p30_0p15_512_p999_hard50k_dueling_h_4d_rel373220.pt'):
    """ Evaluate a trained model and compute your leaderboard scores

	NO CHANGES SHOULD BE MADE TO THIS FUNCTION

    Parameters
    -------
    env: Carla Env
        environment to evaluate on
    load_path: str
        path to load the model (.pt) from
    """
    episode_rewards = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]

    # Build & load network
    # policy_net = DQN(action_size, device).to(device)
    policy_net = DDQNLidar(action_size, device).to(device)
    checkpoint = torch.load(load_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for i in range(10):
        env.seed(seeds[i])
        obs, done = env.reset(1000000, False), False
        obs = get_state(obs)
        t = 1000000

        # Run each episode until episode has terminated or 600 time steps have been reached
        episode_rewards.append(0.0)
        while not done and t < 1000128:
            action_id = select_greedy_action(obs, policy_net, action_size)
            # action = actions[action_id]
            # print(action_id
            obs, rew, done, _ = env.step(action_id, t)
            obs = get_state(obs)
            episode_rewards[-1] += rew
            t += 1
        print('episode %d \t reward %f' % (i, episode_rewards[-1]))

    print('---------------------------')
    print(' total score: %f' % np.mean(np.array(episode_rewards)))
    print('---------------------------')


def behavior_cloning_prelearn(policy_net, device, lr=1e-5, n_actions=4, batch_size=64):
    return train_classification(data_folder="C:/Users/wanga/OneDrive/Documents/BU/Coursework/LimoProject/", save_path="../cloning", 
                               lr=lr, n_actions=n_actions, batch_size=batch_size, policy_net=policy_net, device=device)
# def behavior_cloning_prelearn(lr=1e-5, batch_size=32):
#     return train_classification(lr=lr, batch_size=batch_size)
#3e_5_300ktime_hard_0.9999g_256batch_500kbuff_noher_8d_ddd_
#lr=3e-5,
#          total_timesteps = 200000,
#          buffer_size = 100000,
#          exploration_fraction=0.50,
#          exploration_final_eps=0.10,
#          train_freq=1,
#          action_repeat=3,
#          batch_size=512,
#          learning_starts=5000,
#          gamma=0.9999,
#          target_network_update_freq=5000,
def learn(env,
          lr=1e-5,
          total_timesteps = 100000,
          buffer_size = 10000000,
          exploration_fraction=0.3,
          exploration_final_eps=0.2,
          train_freq=1,
          action_repeat=3,
          batch_size=256,
          learning_starts=10000,
          gamma=0.99,
          target_network_update_freq=20000,
          model_identifier='agent_car_behavior', 
          doubleDQN=False,
          behavior_preclone=False,
          perform_her=False,
          simple=False):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
    print("Actions: ", action_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build networks
    policy_net = DQNLSTMLidar(action_size, device).to(device)#BasicDQN(action_size, device).to(device) #DQNLidar(action_size, device).to(device)

    if behavior_preclone:
        policy_net = behavior_cloning_prelearn(policy_net, device, lr, action_size, batch_size)
        #total_timesteps = int(total_timesteps/4)
        print("Behavior Cloning Done -> going RL")
    
    target_net = DQNLSTMLidar(action_size, device).to(device)#BasicDQN(action_size, device).to(device)#DQNLidar(action_size, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    
    reward_exploration = RewardBasedEGreedy(total_timesteps, exploration_final_eps, 1.0)

    # Initialize environment and get first state
    obs = get_state(env.reset(0, simple))

    trajectory = []

    traj_added = 0

    action_steps = 0
    episodes_performed = 0
    prev_episode_rew = -100000

    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        # if (t == 0):
        #     last_action_id = action_id
        # else:
        #     while (last_action_id == get_opposite_action(action_id)):
        #         #print(action_id, last_action_id, get_opposite_action(action_id))
        #         action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        #     last_action_id = action_id

        #if (action_steps > 0):
        #    print("Boring E: ", exploration.value(t), "FUN EEEEEEE: ", reward_exploration.value(t, prev_episode_rew), "Cur ERew: ", prev_episode_rew)
        #action_id = select_reward_exploratory_action(obs, policy_net, action_size, reward_exploration, t, prev_episode_rew)
        # env_action = actions[action_id]

        # Perform action fram_skip-times
        for f in range(action_repeat):
            action_steps += 1
            new_obs, rew, done, _ = env.step(action_id, t)
            rew = np.clip(rew, -1, 1)

            episode_rewards[-1] += rew
            if done:
                break
            else:
                trajectory.append([obs, action_id, rew, new_obs, float(done)])


        # cv.imshow("Gah", cv.normalize(new_obs, None, 0, 255, cv.NORM_MINMAX))

        # Store transition in the replay buffer.
        # print("obs", obs.shape)

        new_obs = get_state(new_obs)
        # print("new", obs.shape)

        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        traj_added += 1
        obs = new_obs

        if done:
            episodes_performed += 1
            
            print("BufferLen", len(replay_buffer))

            # toss HER experiences into buffer for future sampling
            if perform_her and episodes_performed > 0 and len(trajectory) > 2:
                print("Performing Her")
                #_, _, _, final_state, _ = copy.deepcopy(trajectory[-1])
                
                her_sample_count = 0
                for ti in range(0, len(trajectory), action_repeat):#, int(action_repeat*)):
                    state, action, _, new_state, done = copy.deepcopy(trajectory[ti])
                    
                    if ti + int(action_repeat*4) < len(trajectory)-1:
                        _, _, _, goal_state, _ = copy.deepcopy(trajectory[ti + int(action_repeat*4)])
                        goal_distance = np.linalg.norm(goal_state[20:, 0, :] - state[:, 20:, 0, :].squeeze(0))
                    else:
                        _, _, _, goal_state, _ = copy.deepcopy(trajectory[-1])
                        goal_distance = np.linalg.norm(goal_state[20:, 0, :] - state[:, 20:, 0, :].squeeze(0))


                    
                    # if action < 3: #onyl do this if we are actively moving forward otherwise we just have wiggle samples
                    state, new_state = env.overwrite_state_hers(state, new_state, goal_state)
                    # print("state", state.shape)

                    new_state = get_state(new_state)
                    # print("new_state", new_state.shape)

                    fake_reward = env.her_reward(state, ti, goal_distance)

                    replay_buffer.add(state, action, fake_reward, new_state, float(True))

                    her_sample_count += 1

                print("HER Added: ", her_sample_count, "Traj Added: ", traj_added)
                traj_added = 0
            trajectory.clear()

            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + "\t episode #: " + str(episodes_performed) + "\t reward: " + str(episode_rewards[-1]))
            obs = get_state(env.reset(t, simple))


            if episodes_performed % 150 == 0 and t > learning_starts:
                print(episodes_performed)
                torch.save(policy_net.state_dict(), model_identifier+str(t)+'.pt')
                visualize_training(episode_rewards, training_losses, model_identifier, t)

            #prev_episode_rew = episode_rewards[-1]
            episode_rewards.append(0.0)

            action_steps = 0

        if t > learning_starts and t % train_freq == 0 and len(env.state_sequence) > 2:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, doubleDQN)
            print("Loss @ T: " + str(t) + " -> " + str(loss) )
            training_losses.append(loss)        

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)
        
        

    # Save the trained policy network
    torch.save(policy_net.state_dict(), model_identifier+'done.pt')

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier, t)
