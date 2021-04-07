import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def setup_training (self) -> None:

    self.in_size     = 17
    self.actions     = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    self.agent       = DQN_Agent(self.in_size, self.actions)
    self.target_net  = DQN_Agent(self.in_size, self.actions)
    self.tot_rewards = 0

    # Set training and hyperparameters.
    self.optimizer   = optim.Adam(self.agent.parameters(), lr = 0.001)
    self.criterion   = nn.SmoothL1Loss()
    self.batch_size  = 25
    self.step_opt    = 100 
    self.min_memory  = 4000
    self.max_memory  = 40000 
    self.epsilon     = 0.9
    self.min_epsilon = 0.25

    try:
        self.losses      = np.load('losses.npy')
        self.coll_reward = np.load('collected_rewards.npy')
    except:
        self.losses      = np.zeros(1)
        self.coll_reward = np.zeros(1)

    # Load a pre trained agent if it exist,
    try:
        self.agent.load_state_dict(torch.load('agent'))
        self.target_net.load_state_dict(torch.load('agent'))
    except:
        print('No saved agent found.')

    # Load memory from previous games if it exist.
    try:
        self.memory = np.load('memory.npy')
    except:
        self.memory = np.zeros((1, 36))
    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: str):

    # Load the target network at the very beginning of a new game. 
    try:
        if old_game_state['step'] == 1:
            self.target_net.load_state_dict(torch.load('agent'))

        # Load the new optimized model every few iterrations.
        if old_game_state['step'] % self.step_opt == 0:
            self.agent.load_state_dict(torch.load('agent'))
    except:
        return 0

    ### PROCESS THE OLD GAME STATE ###
    # Obtain the old state of the game.
    arena = old_game_state['field']
    agent = np.array(old_game_state['self'][3])
    coins = np.array(old_game_state['coins'])
    step  = old_game_state['step']

    # Choose the five clossest coins to present to the network.
    coin_vector = 1000 * np.ones(10)
    distances   = np.sum(np.abs(coins - agent),1).reshape(-1,1)
    coins       = np.hstack((coins, distances))
    coins       = coins[coins[:,2].argsort()][:5,:2]
    coin_vector[0:coins.flatten().shape[0]] = coins.flatten()

    # Determine possible movements.
    movements = {'UP'  : np.array([-1,  0]), 'DOWN' : np.array([1,0]),
                 'LEFT': np.array([ 0, -1]), 'RIGHT': np.array([0,1])}
    
    up     = movements['UP']    + agent
    down   = movements['DOWN']  + agent
    left   = movements['LEFT']  + agent
    right  = movements['RIGHT'] + agent

    moves    = np.ones(4)
    moves[0] = arena[up[0],    up[1]]
    moves[1] = arena[down[0],  down[1]]
    moves[2] = arena[left[0],  left[1]]
    moves[3] = arena[right[0], right[1]]

    # Create the input vector that represents the current state and infer 
    # the action by the agent.
    state  = torch.from_numpy(np.hstack((
                        coin_vector, moves, agent, step)).reshape(1,-1)).float()
    
    #if self.epsilon > np.random.uniform(0,1):
    #    _, i = self.agent.random_action()
    #    action_vector = torch.zeros((1,4))
    #    action_vector[:, int(i)] = 1
    #else:
    #    action_vector = self.agent.forward(state)

    #epsilon = np.exp(-0.01 * step)
    #if epsilon < self.min_epsilon:
    #    epsilon = self.min_epsilon
    #else:
    #    self.epsilon = epsilon

    # Compute the rewards for each possible action.
    rewards = {'WALL': -1, 'MINIMIZED_DISTANCE': 1, 'COLLECTED_COIN': 10}

    reward = np.zeros(4)
    for i, action in enumerate([up, down, left, right]):
        
        if arena[action[0], action[1]] == -1:
            reward[i] = rewards['WALL']
            continue

        for coin in coins:
            if coin[0] == action[0] and coin[1] == action[1]:
                reward[i] += rewards['COLLECTED_COIN']
            
            if np.sum(np.abs(coin - action)) < np.sum(np.abs(coin - agent)):
                reward[i] += rewards['MINIMIZED_DISTANCE']


    ### PROCESS THE NEW GAME STATE ###

    arena = new_game_state['field']
    agent = np.array(new_game_state['self'][3])
    coins = np.array(new_game_state['coins'])
    step  = new_game_state['step']

    # Choose the five clossest coins to present to the network.
    coin_vector = 1000 * np.ones(10)
    distances   = np.sum(np.abs(coins - agent),1).reshape(-1,1)
    coins       = np.hstack((coins, distances))
    coins       = coins[coins[:,2].argsort()][:5,:2]
    coin_vector[0:coins.flatten().shape[0]] = coins.flatten()

    # Determine possible movements.
    movements = {'UP'  : np.array([-1,  0]), 'DOWN' : np.array([1,0]),
                 'LEFT': np.array([ 0, -1]), 'RIGHT': np.array([0,1])}
    
    up     = movements['UP']    + agent
    down   = movements['DOWN']  + agent
    left   = movements['LEFT']  + agent
    right  = movements['RIGHT'] + agent

    moves    = np.ones(4)
    moves[0] = arena[up[0],    up[1]]
    moves[1] = arena[down[0],  down[1]]
    moves[2] = arena[left[0],  left[1]]
    moves[3] = arena[right[0], right[1]]

    # Store the made old state, taken actions, rewards and new state.
    new_state   = np.hstack((coin_vector, moves, agent, step)).reshape(1,-1)
    state_tuple = np.hstack((state.numpy().flatten(), 
                             np.array(self.actions.index(self_action)), 
                             np.array(reward[self.actions.index(self_action)]), 
                             new_state.flatten()))
    
    # Check if the replay memory eached its maximal capacity. If this is the case,
    # the oldest entry in memory will be removed.
    current_capacity = self.memory.shape[0]
    if current_capacity > self.max_memory:
        self.memory = self.memory[1:,:]                                    # Remove the oldest entry.
        self.memory =  np.vstack((self.memory, state_tuple.reshape(1,-1))) 
    else:
        self.memory = np.vstack((self.memory, state_tuple.reshape(1,-1)))

    KDE = nn.KLDivLoss(reduction='sum')

    if step % self.step_opt == 0 and current_capacity >= self.min_memory:

        # Sample random batch from replay memory.
        idx   = np.random.randint(0, self.memory.shape[0], self.batch_size)
        batch = self.memory[idx,:]
        
        # Unpack the sampled batch
        states   = torch.from_numpy(batch[:,:self.in_size]).float()
        actions  = batch[:, self.in_size: self.in_size + 1]
        rewards  = torch.from_numpy(batch[:, self.in_size + 1 : self.in_size + 2]).float().flatten()
        n_states = torch.from_numpy(batch[:, self.in_size + 2 :]).float()

        # Compute q max from the target net.
        pred_q   = self.target_net.forward(n_states).detach()
        q_max, _ = torch.max(pred_q, dim = 1)

        values = np.arange(0,4)
        p_dist = to_histogram(self.memory[-step:, 17], values)
        q_dist = to_histogram(np.random.randint(0,4, p_dist.shape[0]), values)

        p_dist = torch.from_numpy(p_dist)
        q_dist = torch.from_numpy(q_dist)

        # Make an action and Compute the loss for the made actions.´
        q     = self.agent.forward(states)[tuple(np.arange(0, self.batch_size)), tuple(actions.flatten())]
        loss  = self.criterion(rewards + (np.exp(-0.01 * step) * q_max), q) + 0.75 * KDE(p_dist, q_dist).abs()
        
        # Perform the optimization of the agent.
        self.agent.zero_grad()
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
 
        # Store the optimized agent as well as the losses.
        self.losses      = np.hstack((self.losses, loss.detach().numpy()))
        self.coll_reward = np.hstack((self.coll_reward, np.mean(rewards.numpy())))
        torch.save(self.agent.state_dict(), 'agent')
        np.save('memory.npy', self.memory)
        np.save('losses.npy', self.losses)
        np.save('collected_rewards.npy', self.coll_reward)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: str):

    # Save the memory and the trained agent.
    torch.save(self.agent.state_dict(), 'agent')
    np.save('memory.npy', self.memory)
    np.save('losses.npy', self.losses)
    np.save('collected_rewards.npy', self.coll_reward)



class DQN_Agent(nn.Module):

    def __init__ (self, input_dimension: int, actions: list) -> None:
        
        super(DQN_Agent, self).__init__()

        self.input_dimension = input_dimension
        self.actions         = actions

        self.FC1 = nn.Linear(input_dimension, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, 64)
        self.FC5 = nn.Linear(64, len(self.actions))
    
    def forward(self, x):

        x = F.leaky_relu(self.FC1(x))
        x = F.leaky_relu(self.FC2(x))
        x = F.leaky_relu(self.FC3(x))
        x = F.leaky_relu(self.FC4(x))
        return self.FC5(x)
    
    def random_action(self) -> tuple:
        
        i = np.random.randint(0, len(self.actions))
        return self.actions[i], i
    
    def make_action(self, x):
        
        pred = self.forward(x)
        return self.actions[int(pred.argmax())]


def to_histogram(x: np.ndarray, values = np.arange(0,4), normalize: bool = True) -> np.ndarray:
    """ Computes a histogram of a given array.
    
    Arguments:
    ----------
    x: (np.ndarray) containing the samples to compute the histogram from.
    
    Returns:
    --------
    out_histogram: (np.ndarray) histogram.
    """
    # Determine unique values
    unique_values = values.copy()
    
    # Compute the output histogram.
    out_histogram = np.zeros(unique_values.shape[0])
    for i, unique_value in enumerate(unique_values):
        out_histogram[i] = x[x == unique_value].shape[0]
    
    if normalize:
        out_histogram = out_histogram/np.sum(out_histogram)
    
    return out_histogram 
