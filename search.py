"""I am to lazy to search for hyperparameters, so let the computer do it."""

from collections import deque
from datetime import datetime
import os

import numpy as np
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from unityagents import UnityEnvironment

import agents
import config as cfg
from environment import Environment
from training import episode

EARLY_END_NUM_EPISODES = 100
NUM_MAX_EPISODES = 10000


def trainable(hpara):
    pid = os.getpid()

    env_unity = UnityEnvironment(
        file_name=cfg.PATH_TO_TENNIS,
        worker_id=pid,
        seed=datetime.now().microsecond,
        base_port=1,
        no_graphics=True,
    )

    env = Environment(env_unity)

    agent_spaceplayer = agents.SpacePlayerDDPG(
        env.state_space_size,
        env.action_space_size,
        seed=datetime.now().microsecond,
        hpara=hpara,
        add_noise=True,
        batch_size=128,
        buffer_size=int(1e6),
    )

    # Track the mean score
    sliding_window = deque(maxlen=EARLY_END_NUM_EPISODES)
    for i in range(NUM_MAX_EPISODES):

        score, episode_len = episode(env, agent_spaceplayer)

        min_score = np.min(score)
        mean_score = np.mean(score)
        max_score = np.max(score)

        sliding_window.append(max_score)

        # print the current score
        sliding_mean = np.mean(sliding_window)

        yield {
            "sliding_mean": sliding_mean,
            "episodes": i,
            "episode_len": episode_len,
            "max_score": max_score,
            "min_score": min_score,
            "mean_score": mean_score,
        }
    env.close()


bayesopt = BayesOptSearch(metric="mean_score", mode="max")
asha_scheduler = ASHAScheduler(
    time_attr="episodes",
    metric="sliding_mean",
    mode="max",
    max_t=NUM_MAX_EPISODES,
    grace_period=100,
    reduction_factor=3,
    brackets=1,
)
analysis = tune.run(
    trainable,
    config=agents.DDPG.hyperparamter_space(),
    search_alg=bayesopt,
    scheduler=asha_scheduler,
    resources_per_trial={"cpu": 2},
    num_samples=20,
)

print(
    "best config: ",
    analysis.get_best_config(
        metric="mean_score",
        mode="max",
    ),
)
