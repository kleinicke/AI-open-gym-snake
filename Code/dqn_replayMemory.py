"""
    AI Project by Florian Kleinicke
    Q-learning for Snake in an OpenAI/PLE environment
"""

#get the there defined variables
from dqn_snake import REPLAY_MEMORY, BATCH_SIZE

from collections import deque
from collections import namedtuple
from random import sample
import numpy as np
import torch

"""
    Stores the last few Transitions and is able to return a random set on Transitions as trainingsdata.
    Used, so the trainingsdata is less correlated
"""
class ReplayMemory(object):
    def __init__(self, capacity=REPLAY_MEMORY):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state','additional_state', 'action', 'reward', 'next_state','next_additional_state'))
        #self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self._available = False

    def put(self, state: np.array,additional_state: np.array, action: torch.LongTensor, reward: np.array, next_state: np.array,next_additional_state: np.array):

        state = torch.FloatTensor(state)
        additional_state = torch.FloatTensor(additional_state)
        reward = torch.FloatTensor([reward])
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)
            next_additional_state = torch.FloatTensor(next_additional_state)
        transition = self.Transition(state=state,additional_state=additional_state, action=action, reward=reward, next_state=next_state,next_additional_state=next_additional_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def size(self):
        return len(self.memory)

    def is_available(self):
        if self._available:
            return True

        if len(self.memory) > BATCH_SIZE:
            self._available = True
        return self._available
