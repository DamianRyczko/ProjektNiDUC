
from bsc_harq import BscHarqSimulator
from gilbert_elliott_harq import GilbertElliottHarqSimulator
import random
import csv
import yaml

#plik cvs z parametrami (długośc danych, ilośc potwórzeń)

if __name__ == "__main__":

    def read_yaml_config(filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
        return config


    config = read_yaml_config('config.yaml')


    def generate_test_data_csv(length, n_of_samples, writer):
        for _ in range(n_of_samples):
            row = []
            for j in range(length):
                row.append(str(random.choice([0, 1])))
            writer.writerow(row)


    def read_csv_data(filename):
        data = []
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:

                data_stream = []
                for bit in row:
                    data_stream.append(int(bit, 2))

                data_package = []
                for i in range(0, len(data_stream), 4):
                    data_package.append(data_stream[i:i + 4])

                data.append(data_package)

        return data


    if bool(config['should_generate_new_data']):
        length = int(config['length'])
        n_of_samples = int(config['n_of_samples'])
        if length % 4 == 0 and n_of_samples > 0:
            with open("test_data.csv", "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                generate_test_data_csv(length, n_of_samples, writer)
        else:
            print("Error: Length must be divisible by 4 and number of samples must be greater than 0.")

    packet_size = 4  # Assumed 4-bit data (before encoding)
    max_retransmissions = int(config['max_retransmissions']  )
    
    ber = float(config['ber'])
    bsc_simulator = BscHarqSimulator(packet_size, ber, max_retransmissions)

    p_good_to_bad = float(config['p_good_to_bad'])
    p_bad_to_good = float(config['p_bad_to_good'])
    ber_good = float(config['ber_good'])
    ber_bad = float(config['ber_bad'])
    gilbert_simulator = GilbertElliottHarqSimulator(packet_size, p_good_to_bad, p_bad_to_good, ber_good, ber_bad,
                                                    max_retransmissions)
    data = read_csv_data('test_data.csv')


    def simulate(simulator, filename, data):
        with open(filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["length", "successful?", "n_repeated"])
            for row in data:
                data_length = len(row) * 4
                is_successful = True
                total_unsuccessful_transmissions = 0
                for package in row:
                    is_package_successful, package_unsuccessful_transmissions = simulator.simulate(package)
                    total_unsuccessful_transmissions += package_unsuccessful_transmissions
                    if not is_package_successful:
                        is_successful = False
                writer.writerow([data_length, is_successful, total_unsuccessful_transmissions])


    if bool(config['should_simulate_bsc']):
        print("Simulation with BSC Channel:")
        simulate(bsc_simulator, "bsc_simulation.csv", data)

    if bool(config['should_simulate_gilbert_elliott']):
        print("\nSimulation with Gilbert-Elliott Channel:")
        simulate(gilbert_simulator, "gilbert_elliott_simulation.csv", data)

