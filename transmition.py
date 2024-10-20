import numpy as np
import csv
from typing import List, Tuple
from collections import Counter

#add noise to data
def binary_symmetric_channel(data: np.ndarray, bit_error_rate: float) -> np.ndarray:
    noise = np.random.choice([0, 1], size=len(data), p=[1 - bit_error_rate, bit_error_rate])
    return np.bitwise_xor(data, noise)

def gilbert_elliott_channel(data: np.ndarray, p_clear_to_noisy: float, p_noisy_to_clear: float, 
                            ber_clear: float, ber_noisy: float, initial_state: str = "clear") -> np.ndarray:
    state = initial_state
    output = np.zeros_like(data)
    
    for i in range(len(data)):
        #state transition
        if state == "clear":
            state = "noisy" if np.random.random() < p_clear_to_noisy else "clear"
        else:
            state = "clear" if np.random.random() < p_noisy_to_clear else "noisy"
        
        #apply errors based on current state
        current_ber = ber_clear if state == "clear" else ber_noisy
        bit_error = np.random.choice([0, 1], p=[1 - current_ber, current_ber])
        output[i] = data[i] ^ bit_error
    
    return output

def hamming_encode(data: List[int]) -> List[int]:
    if len(data) != 4:
        raise ValueError("Input must be a 4-bit list")

    encoded = [0] * 7
    encoded[2], encoded[4], encoded[5], encoded[6] = data

    #calculate parity bits with xor
    encoded[0] = encoded[2] ^ encoded[4] ^ encoded[6]
    encoded[1] = encoded[2] ^ encoded[5] ^ encoded[6]
    encoded[3] = encoded[4] ^ encoded[5] ^ encoded[6]

    return encoded

def hamming_decode(encoded: List[int]) -> List[int]:
    if len(encoded) != 7:
        raise ValueError("Input must be a 7-bit list")

    #calculate syndrome
    syndrome = [
        encoded[0] ^ encoded[2] ^ encoded[4] ^ encoded[6],
        encoded[1] ^ encoded[2] ^ encoded[5] ^ encoded[6],
        encoded[3] ^ encoded[4] ^ encoded[5] ^ encoded[6]
    ]

    #correct error if present
    error_position = (syndrome[0] * 1) + (syndrome[1] * 2) + (syndrome[2] * 4) - 1
    if error_position >= 0:
        encoded[error_position] ^= 1

    #extract data bits
    return [encoded[2], encoded[4], encoded[5], encoded[6]]

def simulate_harq(package: List[int], channel_func, max_retransmissions: int, **channel_params) -> Tuple[bool, int]:
    chunks = chunk_data(package)
    total_attempts = 0
    for chunk in chunks:
        encoded_data = np.array(hamming_encode(chunk))
        attempts = 0
        while attempts < max_retransmissions:
            transmitted_data = channel_func(encoded_data, **channel_params)
            decoded_data = hamming_decode(transmitted_data.tolist())
            if decoded_data == chunk:
                break
            attempts += 1
        total_attempts += attempts
        if attempts == max_retransmissions:
            return False, total_attempts
    return True, total_attempts

def generate_test_data(data_length: int, num_samples: int) -> List[List[int]]:
    return [np.random.choice([0, 1], size=data_length).tolist() for _ in range(num_samples)]

def save_simulation_results(filename: str, results: List[List]):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["packet_length", "transmission_successful", "num_attempts"])
        writer.writerows(results)

def chunk_data(data: List[int], chunk_size: int = 4) -> List[List[int]]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def run_simulation(data: List[List[int]], channel_func, max_retransmissions: int, **channel_params) -> List[List]:
    results = []
    total_bits = 0
    total_errors = 0
    for packet in data:
        success, attempts = simulate_harq(packet, channel_func, max_retransmissions, **channel_params)
        
        #compare input packet with the result of transmission
        decoded_packet = []
        for chunk in chunk_data(packet):
            encoded_chunk = hamming_encode(chunk)
            transmitted_chunk = channel_func(np.array(encoded_chunk), **channel_params)
            decoded_chunk = hamming_decode(transmitted_chunk.tolist())
            decoded_packet.extend(decoded_chunk)
        
        #count bit errors
        errors = sum(a != b for a, b in zip(packet, decoded_packet))
        total_bits += len(packet)
        total_errors += errors
        
        results.append([len(packet), success, attempts, errors])
    
    error_rate = total_errors / total_bits
    return results, error_rate

def print_retransmission_stats(results: List[List]):
    retransmission_counts = Counter(result[2] for result in results)
    total_transmissions = len(results)
    
    print("| Number retransmissions | number / all  | %of time |")
    print("|------------------------|---------------|----------|")
    
    for retransmissions in range(max(retransmission_counts.keys()) + 1):
        count = retransmission_counts[retransmissions]
        percentage = (count / total_transmissions) * 100
        print(f"| {retransmissions:^22} | {count:^5} / {total_transmissions:<5} | {percentage:^7.3f}% |")

def main():
    save_to_csv = 0

    #simulation parameters
    data_length = 1024  #data stream length
    num_samples = 256   #number of streams
    max_retransmissions = 5

    #generate test data
    test_data = generate_test_data(data_length, num_samples)

    #BSC simulation
    ber_bsc = 0.025
    bsc_results, finall_ber = run_simulation(test_data, binary_symmetric_channel, max_retransmissions, bit_error_rate=ber_bsc)
    
    if save_to_csv == 1:
        save_simulation_results('bsc_simulation_results.csv', bsc_results)
        print("BSC simulation completed. Results saved in 'bsc_simulation_results.csv' (max 256 examples)")
    
    print("BSC Simulation Results:")
    print_retransmission_stats(bsc_results)
    print()

    #Gilbert-Elliott simulation
    ber_clear = 0.001
    ber_noisy = 0.025
    p_clear_to_noisy = 0.1
    p_noisy_to_clear = 0.3
    ge_params = {
        'p_clear_to_noisy': p_clear_to_noisy,
        'p_noisy_to_clear': p_noisy_to_clear,
        'ber_clear': ber_clear,
        'ber_noisy': ber_noisy
    }
    ge_results, ge_final_ber = run_simulation(test_data, gilbert_elliott_channel, max_retransmissions, **ge_params)
    
    if save_to_csv:
        save_simulation_results('gilbert_elliott_simulation_results.csv', ge_results)
        print("Gilbert-Elliott simulation completed. Results saved in 'gilbert_elliott_simulation_results.csv'")
    
    print("Gilbert-Elliott Simulation Results:")
    print_retransmission_stats(ge_results)
    print(f"\nError rate of whole transmission: {ge_final_ber:.6f}")

if __name__ == "__main__":
    main()
