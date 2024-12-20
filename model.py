import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def actions_to_classes(actions, num_actions):
    """
    For a given set of actions map every action to its corresponding
    action-class representation. Assume there are C different classes, then
    every action is represented by a C-dim vector which has exactly one
    non-zero entry (one-hot encoding). That index corresponds to the class
    number.
    actions:        python list of N torch.Tensors of size 3
    return          python list of N torch.Tensors of size C
    """
    one_hots = []
    for action in actions:
        oneHot = [0] * num_actions
        oneHot[action] = 1
        one_hots.append(oneHot)
    
    return one_hots

class BasicDQN(nn.Module):
    def __init__(self, action_size, device):
        super().__init__()
        
        self.height = 1
		
        self.device = device 
        self.action_size = action_size
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, action_size)


    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            #print(observation.shape)
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)
            observation = observation[:, :, :self.height, :]
        #print(observation)
        fc1 = torch.relu(self.fc1(self.flat(observation)))
        Q = self.fc2(fc1)
      
        return Q

class DDQNLidar(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 20    
        self.height = 20
        self.depth = 2

        self.c1 = nn.Conv2d(2, 16, kernel_size=2, stride=2) #16 x 10 x 10
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 64, kernel_size=2, stride=2) # 64 x 5 x 5
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(64, 256, kernel_size=5, stride=1) # 256 x 1 x 1

        #results in 1x1x128

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(256, 32768)
        self.fc2 = nn.Linear(32768, 8192)
        self.fc3 = nn.Linear(8192, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)
            observation = observation[:, :, :self.height, :]

        c1 = torch.relu(self.c1(observation))
        c2 = torch.relu(self.c2(c1))
        c3 = torch.relu(self.c3(c2))

        flat = self.flat(c3)

        fc1 = torch.relu(self.fc1(flat))
        fc2 = torch.relu(self.fc2(fc1))
        fc3 = torch.relu(self.fc3(fc2))
        V = self.V(fc3).expand(fc3.size(0), self.action_size)
        A = self.A(fc3)

        Q = V + A - A.mean(1).unsqueeze(1).expand(fc3.size(0), self.action_size)

        return Q
    
class DQNLidar(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 20    
        self.height = 20
        self.depth = 2

        self.c1 = nn.Conv2d(2, 16, kernel_size=2, stride=2) #16 x 10 x 10
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, kernel_size=2, stride=2) # 32 x 5 x 5
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 128, kernel_size=5, stride=1) # 128 x 1 x 1

        #results in 1x1x128

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128, 5096)
        self.fc2 = nn.Linear(5096, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.q_vals = nn.Linear(64, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)
            observation = observation[:, :, :self.height, :]

        c1 = torch.relu(self.c1(observation))
        c2 = torch.relu(self.c2(c1))
        c3 = torch.relu(self.c3(c2))

        flat = self.flat(c3)

        fc1 = torch.relu(self.fc1(flat))
        fc2 = torch.relu(self.fc2(fc1))
        fc3 = torch.relu(self.fc3(fc2))
        out = self.q_vals(fc3)

        return out
    
class DQNLSTMLidar(torch.nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 20    
        self.height = 20
        self.depth = 6

        self.c1 = nn.Conv2d(2, 16, kernel_size=2, stride=2) #16 x 10 x 10
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, kernel_size=2, stride=2) # 32 x 5 x 5
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 128, kernel_size=5, stride=1) # 128 x 1 x 1

        self.lstm = nn.LSTM(128, 64, 1, batch_first=True)


        #results in 1x1x128

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(64, 16)
        self.q_vals = nn.Linear(16, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        
        hidden = None

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            #b, 
            # print(observation.shape)
            observation = torch.from_numpy(observation).to(self.device).permute(0, 1, 4, 2, 3)
            observation = observation[:, :, :, :self.height, :]

        #print(observation.shape)
        for t in range(observation.size(1)):
           # print(observation[:, t, :, :, :].shape)
            c1 = torch.relu(self.c1(observation[:, t, :, :, :]))
            c2 = torch.relu(self.c2(c1))
            c3 = torch.relu(self.c3(c2))

            flat = self.flat(c3)
            #print(flat.shape)
            out, hidden = self.lstm(flat, hidden)
        ##print(out.shape)
        fc1 = torch.relu(self.fc1(out))
        out = self.q_vals(fc1)
       # print(out.shape)
        return out
    
class DQNSEQLidar(torch.nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 20    
        self.height = 20
        self.depth = 6

        self.c1 = nn.Conv2d(6, 24, kernel_size=2, stride=2) #24 x 10 x 10
        self.bn1 = nn.BatchNorm2d(24)
        self.c2 = nn.Conv2d(24, 96, kernel_size=2, stride=2) # 96 x 5 x 5
        self.bn2 = nn.BatchNorm2d(96)
        self.c3 = nn.Conv2d(96, 384, kernel_size=5, stride=1) # 384 x 1 x 1



        #results in 1x1x128

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(384, 5096)
        self.fc2 = nn.Linear(5096, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.q_vals = nn.Linear(64, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)
            observation = observation[:, :, :self.height, :]

        c1 = torch.relu(self.c1(observation))
        c2 = torch.relu(self.c2(c1))
        c3 = torch.relu(self.c3(c2))

        flat = self.flat(c3)

        fc1 = torch.relu(self.fc1(flat))
        fc2 = torch.relu(self.fc2(fc1))
        fc3 = torch.relu(self.fc3(fc2))
        out = self.q_vals(fc3)

        return out
    
class DQNLidar(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 20    
        self.height = 20
        self.depth = 2

        self.c1 = nn.Conv2d(2, 16, kernel_size=2, stride=2) #16 x 10 x 10
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, kernel_size=2, stride=2) # 32 x 5 x 5
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 128, kernel_size=5, stride=1) # 128 x 1 x 1

        #results in 1x1x128

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128, 5096)
        self.fc2 = nn.Linear(5096, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.q_vals = nn.Linear(64, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            #b, h, w, c
            #b, c, h, w 
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)
            observation = observation[:, :, :self.height, :]

        c1 = torch.relu(self.c1(observation))
        c2 = torch.relu(self.c2(c1))
        c3 = torch.relu(self.c3(c2))

        flat = self.flat(c3)

        fc1 = torch.relu(self.fc1(flat))
        fc2 = torch.relu(self.fc2(fc1))
        fc3 = torch.relu(self.fc3(fc2))
        out = self.q_vals(fc3)

        return out

class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.width = 128    
        self.height = 128
        self.depth = 1

        self.c1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)   

        #results in 40x30x64

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(8192, 256)

        self.q_vals = nn.Linear(256, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        if isinstance(observation, torch.Tensor):
            print("Is tensor")
        else:
            observation = torch.from_numpy(observation).to(self.device).permute(0, 3, 1, 2)

        c1 = torch.relu(self.bn1(self.c1(observation)))
        c2 = torch.relu(self.bn2(self.c2(c1)))
        c3 = torch.relu(self.bn3(self.c3(c2)))

        flat = self.flat(c3)

        fc1 = torch.relu(self.fc1(flat))
        out = self.q_vals(fc1)

        return out
    
