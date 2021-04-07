import numpy as np 
import torch as tr 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class DQN_Agent (nn.Module):

    def __init__ (self, input_dimension: int, actions: list):

        self.input_dimension = input_dimension
        self.actions         = actions

        # Define the actula model.
        self.input_layer = nn.Linear(self.input_dimension, 64)
        self.FC1_layer   = nn.Linear(64, 64)
        self.FC2_layer   = nn.Linear(64, 64)
        self.FC3_layer   = nn.Linear(64, 64)
        self.out_layer   = nn.Linear(64, len(self.actions))
    

    def forward (self, x):

        x = F.leaky_relu(self.input_layer(x))
        x = F.leaky_relu(self.FC1_layer(x))
        x = F.leaky_relu(self.FC2_layer(x))
        x = F.leaky_relu(self.FC3_layer(x))
        x = self.out_layer(x)
        return x

class memory:

    def __init__ (self, capacity: int):
        self.capacity = capacity
        self.memory   = list()


    def __call__ (self, state: tuple):

        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]
            self.memory.append(state)
        else:
            self.memory.append(state)
    
    def sample (self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    
    def save(self, file_name):
        pass

    def load(self, file_name):
        pass