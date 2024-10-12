import numpy as np


class HammingCodec:
    def __init__(self):
        pass

    def encode(self, data):
        k = len(data)
        r = 0
        while (2 ** r < k + r + 1):
            r += 1

        total_bits = k + r
        hamming_code = [0] * total_bits

        # Place the data bits in the Hamming code
        data_index = 0
        for i in range(1, total_bits + 1):
            if (i & (i - 1)) == 0:  # If i is a power of 2, it's a parity bit position
                continue
            hamming_code[i - 1] = data[data_index]
            data_index += 1

        # Calculate the parity bits
        for i in range(r):
            parity_pos = 2 ** i  # Position of the parity bit
            parity_value = 0
            for j in range(1, total_bits + 1):
                if j & parity_pos:  # If j has the parity bit position bit set
                    parity_value ^= hamming_code[j - 1]  # XOR with the bit value

            hamming_code[parity_pos - 1] = parity_value

        return hamming_code  # Return as list of integers

    def decode(self, hamming_code):
        # Check if the Hamming code is valid
        if len(hamming_code) != 7 or any(bit not in [0, 1] for bit in hamming_code):
            raise ValueError("Input must be a list of 7 binary integers (0 or 1).")

        # Calculate the number of parity bits
        r = 3  # For 7-bit Hamming code, there are always 3 parity bits

        # Check for errors
        error_position = 0
        for i in range(r):
            parity_pos = 2**i  # Position of the parity bit
            parity_value = 0
            for j in range(1, 8):
                if j & parity_pos:
                    parity_value ^= hamming_code[j - 1]  # XOR with the bit value

            if parity_value != 0:
                error_position += parity_pos

        # Correct the error if there's one
        if error_position != 0:
            print(f"Error detected at position: {error_position}")
            # Flip the bit at the error position (1-indexed)
            hamming_code[error_position - 1] ^= 1  # Flip the detected error bit

        # Extract the original data bits
        data = []
        for i in range(1, 8):
            if (i & (i - 1)) != 0:
                data.append(int(hamming_code[i - 1]))

        return data
    
