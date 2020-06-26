import random


class Transition(object):

    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward


class ReplayMemory(object):

    def __init__(self, capacity, initial_replay_memory_size):
        self.initial_replay_memory_size = initial_replay_memory_size
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """
        Saves a transition in the replay-memory
        """
        transition = Transition(state, action, next_state, reward)
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

    def get_allocated_memory(self):
        total_memory = 0.0
        for transition in self.memory:
            state = transition.state
            next_state = transition.next_state
            action = transition.action
            reward = transition.reward

            # Compute allocated memory of state
            allocated_state = state.nelement() * state.element_size()
            
            if next_state is not None:
                allocated_next_state = next_state.nelement() * next_state.element_size()
            else:
                allocated_next_state = 0

            allocated_action = action.nelement() * action.element_size()
            allocated_reward = reward.nelement() * reward.element_size()

            total_memory += (allocated_state + allocated_next_state + allocated_action + allocated_reward) * 1e-9

        return total_memory
            
