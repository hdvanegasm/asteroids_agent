from collections import deque

import cv2
import gym
import numpy

import torch
import torchvision.transforms as transforms


class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self.obs_buffer = numpy.zeros((2,) + env.observation_space.shape, dtype=numpy.uint8)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            if i == self.skip - 1:
                self.obs_buffer[1] = observation
            elif i == self.skip - 2:
                self.obs_buffer[0] = observation

            total_reward += reward

            if done:
                break

        # Max pooling
        complete_observation = self.obs_buffer.max(axis=0)

        return complete_observation, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RewardClipWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return (numpy.sign(reward), reward)


class FrameTransformWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        if self.grayscale:
            num_colors = 1
        else:
            num_colors = 3

        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(self.height, self.width, num_colors),
                                                dtype=numpy.uint8)

    def observation(self, observation):
        # Transform to grayscale
        if self.grayscale:
            frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            frame = observation

        # Crop image
        frame = cv2.resize(frame,
                           (self.height, self.width),
                           interpolation=cv2.INTER_AREA)

        # Return in format Batch x Channel x Height x Width
        return frame


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k  # Number of stacked frames
        self.frames = deque([], maxlen=self.k)  # Frame buffer
        shape_space = env.observation_space.shape

        # Modify the observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape_space[:-1] + (shape_space[-1] * self.k,)),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for i in range(self.k):
            self.frames.append(obs)
        state = numpy.array(self.get_obs())
        state = numpy.expand_dims(state, axis=0)
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        state = numpy.array(self.get_obs())
        state = numpy.expand_dims(state, axis=0)
        return state, reward, done, info

    def get_obs(self):
        assert len(self.frames) == self.k
        return list(self.frames)


class ScaleFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=numpy.float32(0),
            high=numpy.float32(1),
            shape=self.env.observation_space.shape,
            dtype=numpy.float32
        )

    def observation(self, observation):
        return numpy.array(observation).astype(numpy.float) / 255.0


def transform_image(image):
    return transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.Resize((84, 84)),
        transforms.CenterCrop(84),
        transforms.ToTensor()
    ])(image)[1].unsqueeze(0)


class TransformWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=4, width=84, height=84):
        gym.Wrapper.__init__(self, env)
        self.n_frames_needed = n_frames * 2
        self.width = width
        self.height = height
        self.frames = deque([], maxlen=self.n_frames_needed)

        self.observation_space = gym.spaces.Box(
            low=numpy.float32(0),
            high=numpy.float32(1),
            shape=(1, n_frames, width, height),
            dtype=numpy.float32
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        transformed_obs = transform_image(obs)
        self.frames.append(transformed_obs)
        transformed_state = self.process_state().numpy()
        return transformed_state, reward, done, info

    def reset(self):
        obs = transform_image(self.env.reset())
        for i in range(self.n_frames_needed - 1):
            self.frames.append(torch.zeros((1, self.height, self.width)).detach())
        self.frames.append(obs)

        return self.process_state().numpy()

    def process_state(self):
        last_images = self.frames
        processed_images = []

        for i in range(0, len(last_images), 2):
            first_image = last_images[i]
            second_image = last_images[i + 1]
            join_image = torch.max(first_image, second_image)
            processed_images.append(join_image)

        return torch.cat(processed_images, dim=0).unsqueeze(0)


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_lives = 0

    def reset(self):
        self.current_lives = self.env.unwrapped.ale.lives()
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.unwrapped.ale.lives() < self.current_lives:
            clipped_reward = numpy.sign(-1) * 1.0
        else:
            clipped_reward = numpy.sign(reward) * 1.0
        self.current_lives = self.env.unwrapped.ale.lives()
        return obs, (clipped_reward, reward), done, info


def make_env(env_id, clip_rewards=True, frame_stack=True, scale=True, own_form=False):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id

    if own_form:
        env = TransformWrapper(env)
        env = RewardWrapper(env)
    else:
        env = MaxAndSkipWrapper(env, skip=4)
        env = FrameTransformWrapper(env, width=84, height=84, grayscale=True)
        if clip_rewards:
            env = RewardWrapper(env)
        if frame_stack:
            env = FrameStackWrapper(env, k=4)
        if scale:
            env = ScaleFrameWrapper(env)

    return env
