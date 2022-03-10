"""Code to ease access of the unity env within a brain.

This allows IMHO to write a bit nicer code in the Udacity exercises.
"""

from unityagents import UnityEnvironment


class Environment:
    def __init__(self, env: UnityEnvironment):
        """Wrapper to bin env and brain.

        Args:
            env: A UnityEnvironment.
            brain: The name of the brain.
        """

        self.env = env
        self.brain_name = env.brain_names[0]

        env_info = env.reset(train_mode=False)[self.brain_name]
        self.state_space_size = len(env_info.vector_observations[0])

        self.brain = self.env.brains[self.brain_name]
        self.action_space_size = self.env.brains[
            self.brain_name
        ].vector_action_space_size

    def reset(self, train_mode=True):

        return self.env.reset(train_mode=train_mode)[self.brain_name]

    def step(self, action):

        return self.env.step(action)[self.brain_name]

    def close(self):
        self.env.close()
