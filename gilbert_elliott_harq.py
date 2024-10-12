import numpy as np
from fec_hamming_encode import HammingCodec
from gilbert_elliott_channel import GilbertElliottChannel


class GilbertElliottHarqSimulator:
    def __init__(self, packet_size, p_good_to_bad, p_bad_to_good, ber_good, ber_bad, max_retransmissions):
        self.packet_size = packet_size
        self.p_good_to_bad = p_good_to_bad
        self.p_bad_to_good = p_bad_to_good
        self.ber_good = ber_good
        self.ber_bad = ber_bad
        self.max_retransmissions = max_retransmissions
        self.codec = HammingCodec()
        self.channel = GilbertElliottChannel(p_good_to_bad, p_bad_to_good, ber_good, ber_bad)

    def simulate(self, package):
        print(f"Original Data: {package}")

        encoded_data = self.codec.encode(package)  # FEC encoding
        print(f"Encoded Data: {encoded_data}")

        attempts = 0
        while attempts < self.max_retransmissions:
            print(f"Transmission Attempt: {attempts + 1}")
            transmitted_data = self.channel.step(encoded_data)  # Transmit via Gilbert-Elliott
            print(f"Received Data: {transmitted_data}")

            # Decode and correct errors
            decoded_data = self.codec.decode(transmitted_data)
            print(f"Decoded Data: {decoded_data}")

            # Check if data is correct
            if np.array_equal(decoded_data, package):
                print("Data received correctly!")
                return True, attempts
            else:
                print("Error detected. Requesting retransmission...")

            attempts += 1

        if attempts == self.max_retransmissions:
            print("Transmission failed after maximum retries.")

        return False, attempts
