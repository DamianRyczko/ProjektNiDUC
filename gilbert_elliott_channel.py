import numpy as np


class GilbertElliottChannel:
    def __init__(self, p_good_to_bad, p_bad_to_good, ber_good, ber_bad):
        self.state = "good"  # Initial state is "good"
        self.p_good_to_bad = p_good_to_bad
        self.p_bad_to_good = p_bad_to_good
        self.ber_good = ber_good
        self.ber_bad = ber_bad

    def step(self, package):
        # State transition based on probabilities
        if self.state == "good" and np.random.rand() < self.p_good_to_bad:
            self.state = "bad"
        elif self.state == "bad" and np.random.rand() < self.p_bad_to_good:
            self.state = "good"

        # Apply BER depending on the state
        if self.state == "good":
            return self.apply_bsc(package, self.ber_good)
        else:
            return self.apply_bsc(package, self.ber_bad)

    def apply_bsc(self, package, ber):
        noise = np.random.choice([0, 1], size=len(package), p=[1 - ber, ber])
        return np.bitwise_xor(package, noise)
