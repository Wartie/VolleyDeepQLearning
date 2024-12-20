import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_state(state): 
    """ Helper function to transform state """ 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)

def visualize_training(episode_rewards, training_losses, model_identifier, t):
    """ Visualize training by creating reward + loss plots
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    model_identifier: string
        identifier of the agent
    """
    # print(episode_rewards)
    # print(training_losses)
    if len(episode_rewards) % 15 != 0:
        episodes = np.pad(np.array(episode_rewards), (0, (15 - len(episode_rewards) % 15)), 'constant', constant_values=(0, 0))
        episodes = np.array(episodes).reshape(-1, 15)
    else:
        episodes = np.array(episode_rewards).reshape(-1, 15)

    if len(training_losses) % 100 != 0:
        losses = np.pad(np.array(training_losses), (0, (100 - len(training_losses) % 100)), 'constant', constant_values=(0, 0))
        losses = np.array(losses).reshape(-1, 100)
    else:
        losses = np.array(training_losses).reshape(-1, 100)

    print(episodes)
    print(losses)

    average_rewards = np.mean(episodes, axis = 1)
    average_losses = np.mean(losses, axis = 1)

    print(average_rewards)
    print(average_losses)

    plt.plot(average_rewards)
    plt.savefig("episode_rewards-"+model_identifier+str(t)+".png")
    plt.close()
    plt.plot(average_losses)
    plt.savefig("training_losses-"+model_identifier+str(t)+".png")
    plt.close()

    

    # plt.plot(np.array(episode_rewards))
    # plt.savefig("episode_rewards-"+model_identifier+str(t)+".png")
    # plt.close()
    # plt.plot(np.array(training_losses))
    # plt.savefig("training_losses-"+model_identifier+str(t)+".png")
    # plt.close()

