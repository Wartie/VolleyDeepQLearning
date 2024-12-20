import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import dataset as ds
import matplotlib.pyplot as plt
import model as net

def train_classification(data_folder, save_path, lr, n_actions, batch_size, policy_net, device):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)

    nr_epochs = 75
    batch_size = batch_size
    start_time = time.time()

    n_classes = n_actions
    train_loader = ds.get_dataloader(data_dir=data_folder, batch_size=batch_size)

    losses = []
    
    torch.set_grad_enabled(True)

    
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in_scan, batch_gt = batch[0].to(device), batch[1].to(device)
            #print("gd")
            batch_out = policy_net(batch_in_scan)
            #print("inf")
            soft_max = F.softmax(batch_out, dim=1)
            #print(Q_as_class)
            #print(n_classes)
            batch_gt = net.actions_to_classes(batch_gt, num_actions=n_classes)
            
            loss = cross_entropy_loss(soft_max, batch_gt)
            #loss.requires_grad = True
            #print("loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            #print("back")

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        lrBefore = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        lrAfter = optimizer.param_groups[0]["lr"]

        print("Epoch %5d\t[Train]\tloss: %.6f\tlrb: %.8f\tlra: %.8f \tETA: +%fs" % (
            epoch + 1, total_loss, lrBefore, lrAfter, time_left))

        losses.append(total_loss)
        if total_loss <= 0.01:
            break

    # for batch_idx, batch in enumerate(valid_loader):
    #     batch_in_scan, batch_in_obs, batch_gt = batch[0][0].to(gpu), batch[0][1].to(gpu), batch[1].to(gpu)
    #     batch_out = policy_net(batch_in_scan, batch_in_obs)
    #     batch_gt = policy_net.actions_to_classes(batch_gt)


    cpuLoss = [loss.cpu().detach().float() for loss in losses]

    
    torch.save(policy_net, save_path + "classmodel.pth")
    epochs = list(range(nr_epochs))
    print(epochs)
    print(cpuLoss)
    plt.plot(epochs, cpuLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Training Loss")
    plt.savefig('training_loss_class.png')
    plt.show()

    return policy_net

def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """

    #loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
    batch_gt = torch.tensor(batch_gt).cuda()
    lpred = torch.log(batch_out)
    ytruelogpred = torch.mul(batch_gt, lpred)
    loss_tensor = -torch.mean(torch.sum(ytruelogpred, dim=1))

    return loss_tensor


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, doubleDQN):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss 
        8. Clip the gradients
        9. Optimize the model
    """

    transitions = replay_buffer.sample(batch_size)
    
    obses_t = transitions[0]
    actions = transitions[1]
    rewards = transitions[2]
    obses_tp1 = transitions[3]
    dones = transitions[4]

    
    # print(tensor_obses_tp1.shape)

    non_final_next_states = obses_tp1
    
    Q_vals = policy_net(obses_t)
    # print(torch.from_numpy(actions).to(device).unsqueeze(-1))
    Q_vals = Q_vals.gather(1, torch.from_numpy(actions).to(device).unsqueeze(-1))


    if doubleDQN:
        with torch.no_grad():
            next_actions = policy_net(obses_tp1).argmax(axis=1, keepdim=True)
            next_target_state_values = target_net(obses_tp1).gather(1, next_actions).squeeze()

            Q_vals_tp1 = torch.from_numpy(1 - dones).to(device).unsqueeze(-1) * (next_target_state_values * gamma).unsqueeze(-1) + torch.from_numpy(rewards).to(device).unsqueeze(-1)
    else:
        with torch.no_grad():
            next_state_values = target_net(obses_tp1).max(1).values
            
            Q_vals_tp1 = torch.from_numpy(1 - dones).to(device).unsqueeze(-1) * (next_state_values * gamma).unsqueeze(-1) + torch.from_numpy(rewards).to(device).unsqueeze(-1)

    criterion = torch.nn.HuberLoss()
    # print(Q_vals.shape, Q_vals_tp1.shape, next_state_values.shape, torch.from_numpy(rewards).to(device).unsqueeze(-1).shape)
    # print(Q_vals_tp1.shape, Q_vals_tp1)
    # print(Q_vals.shape, Q_vals_tp1.shape)
    loss = criterion(Q_vals, Q_vals_tp1)
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)

    optimizer.step()

    return loss.cpu().detach().float().item()


# def update_target_net(policy_net, target_net):
    # """ Update the target network
    # Parameters
    # -------
    # policy_net: torch.nn.Module
        # policy Q-network
    # target_net: torch.nn.Module
        # target Q-network
    # """

    # target_net_state_dict = target_net.state_dict()
    # policy_net_state_dict = policy_net.state_dict()

    # for key in policy_net_state_dict:
        # target_net_state_dict[key] = policy_net_state_dict[key] * 0.0001 + target_net_state_dict[key] * (1 - 0.0001)

    # target_net.load_state_dict(target_net_state_dict)


def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    target_net.load_state_dict(policy_net_state_dict)

