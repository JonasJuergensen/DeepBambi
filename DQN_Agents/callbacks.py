import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import random

#from model import (DQN_Agent, memory)

def setup(self):

    self.train   = False
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    self.agent   = DQN_Agent(17, self.actions).float()
    self.epsilon     = 0.9
    self.min_epsilon = 0.1 

    try:
        self.agent.load_state_dict(torch.load('pre_trained_agent'))
        print('Loaded Sucessfully')
    except:
        pass

    

def act(self, game_state: dict) -> str:

    # Obtain the current state of the game.
    arena = game_state['field']
    agent = np.array(game_state['self'][3])
    coins = np.array(game_state['coins'])
    step  = game_state['step']

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

    # Create input vector of coin position, possible movements and agent position.
    if step % 10 == 0:
        try:
            self.agent.load_state_dict(torch.load('agent'))
        except:
            pass
    
    state = torch.from_numpy(np.hstack((coin_vector, moves, agent, step)).reshape(1,-1)).float()
    
    if self.train:
        if self.epsilon > np.random.uniform(0,1):
            _, i = self.agent.random_action()
            action_vector = torch.zeros((1,4))
            action_vector[:, int(i)] = 1
        else:
            action_vector = self.agent.forward(state)
    else:
        action_vector = self.agent.forward(state)
        
    
    epsilon = np.exp(-0.01 * step)
    if epsilon < self.min_epsilon:
        epsilon = self.min_epsilon
    else:
        self.epsilon = epsilon
    
    return self.actions[int(action_vector.argmax())]


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
