import random

import torch.nn.functional
import torch.optim

import utils
import memory
import optim
import test
from network import DeepQNetwork


class Agent(object):
    def __init__(self, env, lr, alpha, momentum, eps, gamma, batch_size, target_update, initial_eps, final_eps,
                 final_time, initial_replay_memory_size, replay_memory_size, periodic_save, state_width,
                 state_height, frames_per_state, n_test_fixed_states, test_epsilon):
        self.gamma = gamma
        self.target_update = target_update
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.final_time = final_time
        self.periodic_save = periodic_save
        self.batch_size = batch_size
        self.n_test_fixed_states = n_test_fixed_states
        self.test_epsilon = test_epsilon

        self.replay_memory = memory.ReplayMemory(replay_memory_size, initial_replay_memory_size)

        # Set device if GPU is available
        # Determines if the capacity of the GPU will works with the memory
        #   - 28672 bytes each screenshot
        #   - 8 screenshots per state
        #   - constants.REPLAY_MEMORY_SIZE is the maximum size of the replay memory
        #   - constants.N_STEPS_FIXED_STATES states * 8 screenshot per state * 1500 bytes per screenshot is the number
        #     of states needed to test the agent
        if (torch.cuda.is_available() and
                (replay_memory_size + self.n_test_fixed_states) * \
                frames_per_state * 28672 < torch.cuda.get_device_properties(torch.device("cuda")).total_memory):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("Using device:", self.device)

        # Generate fixed states
        self.fixed_states = test.get_fixed_states(self, env_id=env.spec.id)

        # Configure CNN
        self.n_actions = env.action_space.n
        self.policy_net = DeepQNetwork(state_height,
                                       state_width,
                                       frames_per_state,
                                       self.n_actions).to(self.device)
        self.target_net = DeepQNetwork(state_height,
                                       state_width,
                                       frames_per_state,
                                       self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.criterion = torch.nn.MSELoss()
        #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, alpha=alpha, momentum=momentum, eps=eps)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0

    def compute_epsilon(self):
        if self.steps_done < 1000000:
            return ((self.final_eps - self.initial_eps) / self.final_time) * self.steps_done + 1
        else:
            return self.final_eps

    def select_action(self, state):
        epsilon_threshold = self.compute_epsilon()
        sample = random.random()
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def select_test_action(self, state):
        epsilon_threshold = self.test_epsilon
        sample = random.random()
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.target_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)

    def optimize(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)

        # Array of True/False if the state is non final
        non_final_mask = torch.tensor(tuple(map(lambda t: t.next_state is not None, transitions)), dtype=torch.bool,
                                      device=self.device)
        non_final_next_states = torch.cat([trans.next_state for trans in transitions
                                           if trans.next_state is not None])
        state_batch = torch.cat([trans.state for trans in transitions])
        action_batch = torch.cat([trans.action for trans in transitions])
        reward_batch = torch.cat([trans.reward for trans in transitions])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).double()

        next_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32).double()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().double()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        #  No clamp (origen del error)
        self.optimizer.step()

        self.steps_done += 1

        return float(loss)



