from collections import deque
import random
from utilities import transpose_list
from collections import namedtuple, deque
import torch
import numpy as np
device = "cpu"

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed = 500):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state","full_state","action", "reward", "next_state","full_next_state", "done"])
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.vstack([e.full_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        full_next_states = torch.from_numpy(np.vstack([e.full_next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return (states,full_states,actions, rewards, next_states,full_next_states, dones)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        full_state = np.concatenate((state[0], state[1]))
        full_next_state = np.concatenate((next_state[0], next_state[1]))
        e = self.experience(state, full_state, action, reward, next_state, full_next_state, done)
        self.memory.append(e)

