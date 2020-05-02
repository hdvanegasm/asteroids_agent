import random

class Transition(object):
    
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
    

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, transition):
        """
        Saves a transition in the replay-memory
        """
        if len(self.memory) <= self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Extract a sample with uniform distribution from the replay-memory
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
        
        