import math
import random

import gym

import torch.optim

from network import DeepQNetwork
import constants
import memory
import utilities

def select_action(state, policy_nn, steps_done, env):
    epsilon_threshold = constants.EPS_END + \
                        (constants.EPS_START - constants.EPS_END) * \
                        math.exp(-1 * steps_done / constants.EPS_DECAY)
    sample = random.sample()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)
    
# TODO terminar optimización
def optimize_model(target_nn, policy_nn, observation, memory):
    if len(memory) < constants.BATCH_SIZE:
        return
    
    transitions = memory.sample(constants.BATCH_SIZE)
    
    non_final_next_state
    
    

def agent():   
    env = gym.make('Asteroids-v0')
    
    n_actions = env.action_space.n
    
    policy_net = DeepQNetwork(constants.STATE_IMG_WIDTH, 
                              constants.STATE_IMG_HEIGHT,
                              n_actions)
    
    target_net = DeepQNetwork(constants.STATE_IMG_WIDTH, 
                              constants.STATE_IMG_HEIGHT,
                              n_actions)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.RMSprop(policy_net.parameters())
    replay_memory = memory.ReplayMemory(constants.REPLAY_MEMORY_SIZE)
    
    cumulative_screenshot = []
    
    for i_episode in range(1):
        observation = env.reset()
        total_reward = 0
        t = 0
        
        for i in range(10):
            env.render(mode='rgb_array').transpose((2, 0, 1))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("T Reward:", total_reward)
                break
            
            t += 1
            
            # Image transformation, resizing, cropping and grayscale       
            observation_tensor = utilities.transform_image(observation)
        
    env.close()


