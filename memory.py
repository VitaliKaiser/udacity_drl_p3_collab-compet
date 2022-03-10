import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory(ABC):
    @abstractmethod
    def sample(self) -> Tuple:
        """Randomly sample a batch of experiences from memory.

        Returns:
            A tuple of "state, action, reward, next_state, done" batched tensors.
        """
        pass

    @abstractmethod
    def add(self, state, action, reward, next_state, done) -> None:
        """Add a new experience to memory."""
        pass

    @abstractmethod
    def ready_to_sample(self) -> bool:
        """Indicator if memory is ready to sampled.

        Returns:
            True if memory is ready to be sampled.
        """
        pass


class LaBER(Memory):
    """Memory with prioritization.

    Based on:
    Lahire, Thibault, Matthieu Geist, and Emmanuel Rachelson. "Large Batch Experience Replay." arXiv preprint arXiv:2110.01528 (2021).
    https://arxiv.org/abs/2110.01528

    Args:
        buffer_size: Maximum size of buffer.
        batch_size: Size of each training batch.
        seed: Random seed.
        batch_size_multiplicator: Multiple of batch_size used to define the size if the "large batch" which is used for rating.
        critic: Critic used for rating experiences in the large batch.
        cast_action_to_long: If true, the returned action tensor is casted to long (for discrete problems).
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seed: int,
        batch_size_multiplicator: int,
        critic,
        cast_action_to_long: bool = False,
    ) -> None:

        self.batch_size = batch_size
        large_batchsize = batch_size * batch_size_multiplicator
        self.memory = ReplayBuffer(
            buffer_size, large_batchsize, seed, cast_action_to_long
        )
        self.critic = critic

    def sample(self) -> Tuple:
        states, actions, rewards, next_states, dones = self.memory.sample()

        self.critic.eval()
        with torch.no_grad():
            q_values = torch.squeeze(self.critic(states, actions))
        self.critic.train()

        # Normalize so we can use q values for sampling
        sampling_weights = q_values - torch.min(q_values)
        sampling_weights = sampling_weights / torch.sum(sampling_weights)

        idx = sampling_weights.multinomial(num_samples=self.batch_size)

        return (
            states[idx],
            actions[idx],
            rewards[idx],
            next_states[idx],
            dones[idx],
        )

    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def ready_to_sample(self) -> bool:
        return self.memory.ready_to_sample()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer(Memory):
    """Fixed-size buffer to store experience tuples.

    Args:
        buffer_size: Maximum size of buffer.
        batch_size: Size of each training batch.
        seed: Random seed.
        cast_action_to_long: If true, the returned action tensor is casted to long (for discrete problems).
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seed: int,
        cast_action_to_long: bool = False,
    ):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.cast_action_to_long = cast_action_to_long

    def add(self, state, action, reward, next_state, done) -> None:
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple:

        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )

        if self.cast_action_to_long:
            actions = (
                torch.from_numpy(
                    np.vstack([e.action for e in experiences if e is not None])
                )
                .long()
                .to(device)
            )
        else:
            actions = (
                torch.from_numpy(
                    np.vstack([e.action for e in experiences if e is not None])
                )
                .float()
                .to(device)
            )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def ready_to_sample(self) -> bool:
        return len(self) >= self.batch_size

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
