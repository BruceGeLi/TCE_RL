from mprl.rl.critic import AbstractCritic


class ValueFunction(AbstractCritic):
    def critic(self, state):
        """
        Evaluate the value of the given state

        Args:
            state: state

        Returns:
            value of the state

        """
        value = self.net(state)
        return value
