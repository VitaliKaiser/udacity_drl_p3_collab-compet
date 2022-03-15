from collections import deque

import numpy as np

# Make tensorboard optional
try:
    from tensorboardX import SummaryWriter

    DEFAULT_WRITER = SummaryWriter()
except:
    DEFAULT_WRITER = None

import enum

from tqdm import tqdm

from agents import Agent
from environment import Environment


class SlidingScoreMode(enum.Enum):
    MIN = enum.auto()
    MEAN = enum.auto()
    MAX = enum.auto()


def episode(env: Environment, agent: Agent, max_length=None, train_mode=True):

    # reset the environment
    env_info = env.reset(train_mode=train_mode)

    # get the current state
    state = env_info.vector_observations

    score = np.zeros(state.shape[0])
    i = 0
    while True:
        action = agent.act(state)

        # Send the action to the environment.
        env_info = env.step(action)

        # Get the next state.
        next_state = env_info.vector_observations
        # Get the reward.
        reward = env_info.rewards
        # See if episode has finished.
        done = env_info.local_done
        # Update the score.
        score += reward

        # Tell the agent about the transition.
        agent.step(state, action, reward, next_state, done)

        # Roll over the state to next time step.
        state = next_state
        i = i + 1
        if np.any(done) or (max_length is not None and i >= max_length):
            break

    # Tell the agent that an episode has ended.
    agent.episode_step()
    return score, i


def eval_episode(env: Environment, agent: Agent):
    # reset the environment
    env_info = env.reset(train_mode=False)

    # get the current state
    state = env_info.vector_observations

    score = np.zeros(state.shape[0])
    i = 0
    while True:
        action = agent.act(state, train_mode=False)

        # Send the action to the environment.
        env_info = env.step(action)

        # Get the next state.
        next_state = env_info.vector_observations
        # Get the reward.
        reward = env_info.rewards
        # See if episode has finished.
        done = env_info.local_done
        # Update the score.
        score += reward

        # Roll over the state to next time step.
        state = next_state
        i = i + 1
        if np.any(done):
            break


def train(
    num_episodes: int,
    env: Environment,
    agent: Agent,
    early_end_sliding_score: float,
    early_end_num_episodes: int,
    sliding_score_mode: SlidingScoreMode = SlidingScoreMode.MEAN,
    max_length_episode=None,
    tensorboard=DEFAULT_WRITER,
):

    # Track the mean score
    scores_window = deque(maxlen=early_end_num_episodes)
    with tqdm(range(num_episodes), desc="training", unit="episode", position=0) as pbar:

        for i in pbar:

            score, ep_length = episode(env, agent, max_length=max_length_episode)

            if score.shape[0] > 1:  # Multiagent case

                scores = {
                    SlidingScoreMode.MIN: np.min(score),
                    SlidingScoreMode.MEAN: np.mean(score),
                    SlidingScoreMode.MAX: np.max(score),
                }
                scores_window.append(scores[sliding_score_mode])
                sliding_mean = np.mean(scores_window)
                pbar.set_postfix(
                    {
                        "max_score": scores[SlidingScoreMode.MAX],
                        "min_score": scores[SlidingScoreMode.MIN],
                        "avg_score": scores[SlidingScoreMode.MEAN],
                        "sliding_mean": sliding_mean,
                        "ep_length": ep_length,
                    }
                )

            else:  # If there is only one agent things can be displayed "easier".
                scores_window.append(score)
                sliding_mean = np.mean(scores_window)
                pbar.set_postfix(
                    {
                        "score": score[0],
                        "sliding_mean": sliding_mean,
                        "ep_length": ep_length,
                    }
                )

            if tensorboard is not None:
                tensorboard.add_scalar("sliding_mean", sliding_mean, i)
                if score.shape[0] > 1:
                    tensorboard.add_scalar("max_score", scores[SlidingScoreMode.MAX], i)
                    tensorboard.add_scalar("min_score", scores[SlidingScoreMode.MIN], i)
                    tensorboard.add_scalar(
                        "avg_score", scores[SlidingScoreMode.MEAN], i
                    )

                else:
                    tensorboard.add_scalar("score", score, i)

                tensorboard.add_scalar("Episode_length", ep_length, i)

                for key, value in agent.summaries():
                    tensorboard.add_scalar(key, value, i)

            if sliding_mean > early_end_sliding_score:
                print(f"Score achieved after {i} episodes.")
                break

    agent.save()
