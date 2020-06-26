import pandas
import torch

import constants
import network
import wrappers

env = wrappers.make_env("BreakoutNoFrameskip-v0")
target_net = network.DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE,
                              env.action_space.n)
target_net.load_state_dict(torch.load("nn_parameters.pth"))
n_total_steps = 100000


q_values = []
while True:
    observation = env.reset()
    t = 0
    while True:
        observation, reward, done, info = env.step(env.action_space.sample())
        observation_tensor = torch.tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            q_value_estimation = target_net(observation_tensor).max(1)[0].view(1, 1).item()
            q_values.append(q_value_estimation)

        if done or len(q_values) == n_total_steps:
            break

    if len(q_values) == n_total_steps:
        break

env.close()

pandas.DataFrame(columns=["q_values"], data=q_values).to_csv("q_values_dist_breakout_rm1m.csv")

