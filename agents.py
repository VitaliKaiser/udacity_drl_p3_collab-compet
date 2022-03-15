import copy
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ray import tune

import memory as mem
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(ABC):
    @abstractmethod
    def act(self, state):
        """Returns actions for given state as per current policy.

        Args:
            state: A tensor with batched states having a shape of [BATCHSIZE, DIM_STATE_SPACE].

        Returns:
            A tensor with actions with shape [BATCHSIZE, DIM_ACTION_SPACE].

        """
        return NotImplemented

    @abstractmethod
    def step(self, state, action, reward, next_state, done) -> None:
        """Informs the client about the step in the environment.

        Args:
            state: A tensor with batched states having a shape of [BATCHSIZE, DIM_STATE_SPACE].
            action: A tensor with batched actions with shape [BATCHSIZE, DIM_ACTION_SPACE].
            reward: A tensor with batched rewards with shape [BATCHSIZE, 1].
            next_state: A tensor with batched next states having a shape of [BATCHSIZE, DIM_STATE_SPACE].
            done: A tensor with batched done flags having a shape of [BATCHSIZE, 1].
        """
        return NotImplemented

    def save(self) -> None:
        """Save all parameters of the model that are needed for inference."""
        pass

    def restore(self) -> None:
        """Load all parameters of the model that are needed for inference."""
        pass

    def episode_step(self) -> None:
        """Informing the agent that a episode has ended."""
        pass

    def summaries(self) -> Dict:
        """Costum metrics of an agent that can be logged on tensorboard."""
        return {}

    @staticmethod
    def hyperparamter_space() -> Dict:
        return {}


def soft_update(local_model, target_model, tau: float):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Args:
        local_model: weights will be copied from
        target_model: weights will be copied to
        tau: interpolation parameter
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


class Random(Agent):
    def __init__(self, action_size) -> None:
        """A random agent as a baseline."""

        super().__init__()
        self.action_size = action_size

    def act(self, state):
        batch_size = state.shape[0]
        return np.random.randint(self.action_size, size=batch_size)

    def step(self, state, action, reward, next_state, done) -> None:
        pass


class RandomContinues(Agent):
    def __init__(self, action_size) -> None:
        """A random agent returning a continues action between -1 and 1."""

        super().__init__()
        self.action_size = action_size

    def act(self, state):
        batch_size = state.shape[0]
        return np.random.uniform(-1.0, 1.0, (batch_size, self.action_size))

    def step(self, state, action, reward, next_state, done) -> None:
        pass


class DQN(Agent):
    """Deep Q-Learning agent supporting discrete state and action-spaces."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hpara: dict,
        buffer_size: int = int(1e5),
        batch_size: int = 64,
    ):
        """Initialize an Agent object.

        Args:
            state_size: dimension of each state
            action_size: dimension of each action
            seed: random seed
            hpara: Hyperparameters defined in hyperparamter_space()
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.update_every = hpara["target_update_interval"]

        # hyperparameter
        self.eps = hpara["eps_start"]
        self.eps_end = max(hpara["eps_start"] - hpara["eps_range"], 0.0)
        self.eps_decay = hpara["eps_decay"]
        self.tau = hpara["tau"]
        self.gamma = hpara["gamma"]

        # Q-Network
        self.qnetwork_local = model.QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = model.QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hpara["lr"])

        # Replay memory
        self.memory = mem.ReplayBuffer(
            buffer_size, batch_size, seed, cast_action_to_long=True
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @staticmethod
    def hyperparamter_space() -> Dict:
        """Hyperparamter space specificed for tune ray.

        Returns dict with:
            "eps_start": starting value of epsilon, for epsilon-greedy action selection.
            "eps_range": eps will move between [eps_start, max(eps_start-eps_range, 0)
            "eps_decay": multiplicative factor (per episode) for decreasing epsilon
            "lr": learning rate for the optimizer
            "tau": interpolation parameter used in soft_update()
            "gammma": discount factor for the rewards
        """
        return {
            "eps_start": tune.uniform(0.0, 1.0),
            "eps_range": tune.uniform(0.0, 1.0),
            "eps_decay": tune.uniform(0.0, 1.0),
            "lr": tune.loguniform(1e-10, 1e-2),
            "tau": 1e-3,  # TODO check if we want to add this to the searchspace
            "gamma": 0.99,  # TODO check if we want to add this to the searchspace
            "target_update_interval": 4,  # TODO check if we want to add this to the searchspace
        }

    def step(self, state, action, reward, next_state, done) -> None:
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                states, actions, rewards, next_states, dones = self.memory.sample()
                self.learn(states, actions, rewards, next_states, dones)

    def episode_step(self) -> None:

        self.eps = max(self.eps * self.eps_decay, self.eps_end)

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, states, actions, rewards, next_states, dones) -> None:
        """Update value parameters using given batch of experience tuples."""

        target_next_actions = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )

        q_targets = rewards + (self.gamma * target_next_actions * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def save(self) -> None:
        torch.save(self.qnetwork_local.state_dict(), "checkpoint.pth")

    def restore(self) -> None:
        self.qnetwork_local.load_state_dict(torch.load("checkpoint.pth"))


class DDPG(Agent):
    """Deep Deterministic Policy Gradient Agents.

    Supports continues spaces and actions."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hpara: dict,
        buffer_size: int = int(1e5),
        batch_size: int = 128,
        add_noise=True,
        memory_type=mem.Types.LABER,
    ):

        self.learn_every_step = hpara["learn_every_step"]
        self.learn_steps = hpara["learn_steps"]
        self.t_step = 0

        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.add_noise = add_noise

        # Hyperparameter
        self.tau = hpara["tau"]
        self.gamma = hpara["gamma"]
        self.noise_level = hpara["noise_level_start"]
        self.noise_level_end = max(
            hpara["noise_level_start"] - hpara["noise_level_range"], 0.0
        )
        self.noise_level_decay = hpara["noise_level_decay"]

        # Actor Network (w/ Target Network)
        self.actor_local = model.Actor(
            state_size, action_size, seed, fc1_units=200, fc2_units=100
        ).to(device)
        self.actor_target = model.Actor(
            state_size, action_size, seed, fc1_units=200, fc2_units=100
        ).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=hpara["lr_actor"]
        )

        # Critic Network (w/ Target Network)
        self.critic_local = model.Critic(
            state_size, action_size, seed, fc1_units=200, fc2_units=100
        ).to(device)
        self.critic_target = model.Critic(
            state_size, action_size, seed, fc1_units=200, fc2_units=100
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=hpara["lr_critic"],
            weight_decay=hpara["weight_decay_critic"],
        )

        # Noise process
        self.noise = OUNoise(
            action_size, seed, sigma=hpara["noise_sigma"], theta=hpara["noise_theta"]
        )

        # Replay memory
        if memory_type == mem.Types.LABER:
            self.memory = mem.LaBER(
                buffer_size,
                batch_size,
                seed,
                batch_size_multiplicator=4,
                critic=self.critic_local,
            )
        else:
            self.memory = mem.ReplayBuffer(buffer_size, batch_size, seed)

    @staticmethod
    def hyperparamter_space():
        """Hyperparamter space specificed for tune ray.

        Returns dict with:
            "tau": interpolation parameter used in soft_update()
            "gammma": discount factor for the rewards
            "lr_actor": learning rate for the actor optimizer
            "lr_critic": learning rate for critic optimizer
            "weight_decay_critic": weight decay for critic optimizer
            "noise_sigma": Sigma parameter of the Ornstein-Uhlenbeck for noise generation.
            "noise_theta": Theta parameter of the Ornstein-Uhlenbeck for noise generation.
            "noise_level_start": Start value of the noise level.
            "noise_level_range": How big is the range between the noise level in the start and end of the training.
            "noise_level_decay": How fast should the noise decay.
            "learn_every_step": Executes learning only every X steps.
            "learn_steps": How many batches should be sampled in every learn step.
        """

        return {
            "tau": tune.loguniform(1e-10, 1e-2),
            "gamma": tune.uniform(0.0, 1.0),
            "lr_actor": tune.loguniform(1e-10, 1e-2),
            "lr_critic": tune.loguniform(1e-10, 1e-2),
            "weight_decay_critic": 0.0,
            "noise_sigma": tune.uniform(0.0, 1.0),
            "noise_theta": tune.uniform(0.0, 1.0),
            "noise_level_start": tune.uniform(0.0, 10.0),
            "noise_level_range": tune.uniform(0.0, 10.0),
            "noise_level_decay": tune.uniform(0.0, 1.0),
            "learn_every_step": 1,
            "learn_steps": 1,
        }

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        if state.dim() < 2:
            state = state.unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample() * self.noise_level
            self.noise_level = max(
                self.noise_level * self.noise_level_decay, self.noise_level_end
            )
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        for i in range(state.shape[0]):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.learn_every_step
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.ready_to_sample():
                for _ in range(self.learn_steps):
                    states, actions, rewards, next_states, dones = self.memory.sample()
                    self.learn(states, actions, rewards, next_states, dones)

    def learn(self, states, actions, rewards, next_states, dones) -> None:
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        soft_update(self.critic_local, self.critic_target, self.tau)
        soft_update(self.actor_local, self.actor_target, self.tau)

        self.noise.reset()

    def save(self) -> None:
        torch.save(self.actor_local.state_dict(), "checkpoint.pth")

    def restore(self) -> None:
        self.actor_local.load_state_dict(torch.load("checkpoint.pth"))

    def episode_step(self) -> None:
        self.noise.reset()


class SpacePlayerDDPG(DDPG):
    """Encodes which players view it is in the state."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hpara: dict,
        buffer_size: int = int(1e5),
        batch_size: int = 128,
        add_noise=True,
    ):

        super().__init__(
            state_size + 1, action_size, seed, hpara, buffer_size, batch_size, add_noise
        )
        self.player_encoding = np.expand_dims(np.array([-1.0, 1.0]), axis=1)

    def act(self, state):

        state = np.concatenate([state, self.player_encoding], axis=1)

        return super().act(state)

    def step(self, state, action, reward, next_state, done):

        state = np.concatenate([state, self.player_encoding], axis=1)
        next_state = np.concatenate([next_state, self.player_encoding], axis=1)

        super().step(state, action, reward, next_state, done)


class MultiAgent(Agent):
    """An agent that just consists of multiple independent agents.

    Args:
        agents: List of agents.
    """

    def __init__(self, agents: List[Agent]) -> None:
        super().__init__()
        self.agents = agents
        self.no_agents = len(agents)  # So we dont have to query this every time

    def act(self, state):

        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(state[i]))
        return np.concatenate(actions)

    def step(self, state, action, reward, next_state, done):

        for i, agent in enumerate(self.agents):
            agent.step(
                state[i : i + 1],
                action[i : i + 1],
                reward[i : i + 1],
                next_state[i : i + 1],
                done[i : i + 1],
            )

    def save(self):
        [agent.save() for agent in self.agents]

    def restore(self):
        [agent.restore() for agent in self.agents]


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.uniform(-1, 1) for i in range(len(x))]
        )
        self.state = x + dx
        return self.state
