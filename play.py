#!/usr/bin/env python3
from unityagents import UnityEnvironment
import numpy as np

import agents
import config as cfg
from environment import Environment

env_unity = UnityEnvironment(file_name=cfg.PATH_TO_TENNIS, worker_id=1)
env = Environment(env_unity)

# Hyperparameter not really relevant for playing/inference.
hpara = {
    "tau": 0.01,
    "gamma": 0.99,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "weight_decay_critic": 0.0,
    "noise_sigma": 1.0,
    "noise_theta": 0.15,
    "learn_every_step": 1,
    "learn_steps": 1,
    "noise_level_start": 5.00,
    "noise_level_range": 4.9,
    "noise_level_decay": 0.99,
}

agent = agents.SpacePlayerDDPG(
    env.state_space_size,
    env.action_space_size,
    seed=123,
    hpara=hpara,
    add_noise=False,
    batch_size=128,
    buffer_size=int(1e6),
)

agent.restore()

for i in range(10):
    state = env.reset(train_mode=False)

    while True:

        action = agent.act(state, train_mode=False)

        # Send the action to the environment.
        env_info = env.step(action)

        # Roll over the state to next time step.
        state = env_info.vector_observations
        i = i + 1
        if np.any(env_info.local_done):
            break
