import numpy as np

class BscChannel:
    def __init__(self, ber):
        self.ber = ber

    def transmit(self, package):
        noise = np.random.choice([0, 1], size=len(package), p=[1 - self.ber, self.ber])
        return np.bitwise_xor(package, noise)
