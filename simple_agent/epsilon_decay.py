from simulator.utilities import *

class EpsilonDecay:
    def __init__(self, epsilon0):
        self.epsilon0 = epsilon0

    def get_epsilon(self, episode):
        pass


class ConstantEpsilon(EpsilonDecay):
    def get_epsilon(self, episode):
        return self.epsilon0


class LinearDecay(EpsilonDecay):
    def __init__(self, max, min, step):
        super().__init__(max)
        self.max = max
        self.min = min
        self.step = step

    def get_epsilon(self, episode):
        if episode > self.step:
            return self.min
        return lerp(self.max, self.min, episode / self.step)


class InvSqrtDecay(EpsilonDecay):
    def get_epsilon(self, episode):
        return self.epsilon0 / math.sqrt(episode + 1)