#!/usr/bin/env python3
from unityagents import UnityEnvironment

import agents
import config as cfg
from environment import Environment
from training import SlidingScoreMode, train

env_unity = UnityEnvironment(
    file_name=cfg.PATH_TO_TENNIS, worker_id=1, no_graphics=True
)
env = Environment(env_unity)

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

agent_spaceplayer = agents.SpacePlayerDDPG(
    env.state_space_size,
    env.action_space_size,
    seed=123,
    hpara=hpara,
    add_noise=True,
    batch_size=128,
    buffer_size=int(1e6),
)


train(
    num_episodes=5000,
    early_end_sliding_score=0.5,
    early_end_num_episodes=100,
    env=env,
    agent=agent_spaceplayer,
    sliding_score_mode=SlidingScoreMode.MAX,
)

env.close()
