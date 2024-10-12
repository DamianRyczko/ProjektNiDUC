# bsc_harq.py
import numpy as np
from fec_hamming_encode import HammingCodec
from bsc_channel import BscChannel


class BscHarqSimulator:
    def __init__(self, packet_size, ber, max_retransmissions):
        self.packet_size = packet_size
        self.ber = ber
        self.max_retransmissions = max_retransmissions
        self.codec = HammingCodec()
        self.channel = BscChannel(ber)

    def simulate(self, package):
        print(f"Original Data: {package}")

        encoded_data = self.codec.encode(package)  # FEC encoding
        print(f"Encoded Data: {encoded_data}")

        encoded_data = np.array(encoded_data)

        attempts = 0
        while attempts < self.max_retransmissions:
            print(f"Transmission Attempt: {attempts + 1}")
            transmitted_data = self.channel.transmit(encoded_data)
            print(f"Received Data: {transmitted_data}")

            decoded_data = self.codec.decode(transmitted_data)
            print(f"Decoded Data: {decoded_data}")

            if np.array_equal(decoded_data, package):
                print("Data received correctly!")
                return True, attempts
            else:
                print("Error detected. Requesting retransmission...")

            attempts += 1

        if attempts == self.max_retransmissions:
            print("Transmission failed after maximum retries.")

        return False, attempts