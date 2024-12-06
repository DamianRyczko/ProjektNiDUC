// test.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>

// Constants for simulation
#define BLOCK_SIZE 256
#define DATA_LENGTH 1024
#define MAX_RETRANSMISSIONS 5
#define HAMMING_DATA_SIZE 4
#define HAMMING_CODE_SIZE 7

// Structure to hold simulation results
struct SimulationResult {
    unsigned long long retransmission_counts[MAX_RETRANSMISSIONS + 1];  // Changed to long long for large iterations
    unsigned long long total_errors;
    unsigned long long total_bits;
};

// Device functions for Hamming code remain the same
__device__ void hamming_encode(const unsigned char* data, unsigned char* encoded) {
    encoded[2] = data[0];
    encoded[4] = data[1];
    encoded[5] = data[2];
    encoded[6] = data[3];
    
    encoded[0] = encoded[2] ^ encoded[4] ^ encoded[6];
    encoded[1] = encoded[2] ^ encoded[5] ^ encoded[6];
    encoded[3] = encoded[4] ^ encoded[5] ^ encoded[6];
}

__device__ bool hamming_decode(unsigned char* encoded, unsigned char* decoded, bool correct_errors) {
    unsigned char syndrome = 0;
    syndrome |= (encoded[0] ^ encoded[2] ^ encoded[4] ^ encoded[6]) << 0;
    syndrome |= (encoded[1] ^ encoded[2] ^ encoded[5] ^ encoded[6]) << 1;
    syndrome |= (encoded[3] ^ encoded[4] ^ encoded[5] ^ encoded[6]) << 2;
    
    if (syndrome != 0) {
        if (!correct_errors) return false;
        encoded[syndrome - 1] ^= 1;
    }
    
    decoded[0] = encoded[2];
    decoded[1] = encoded[4];
    decoded[2] = encoded[5];
    decoded[3] = encoded[6];
    return true;
}

__global__ void init_rng_kernel(curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void bsc_simulation_kernel(
    curandState* states,
    SimulationResult* results,
    float bit_error_rate,
    bool is_error_correcting,
    unsigned int iterations_per_thread  // Added parameter for iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    // Local counters for this thread
    unsigned long long local_retransmissions[MAX_RETRANSMISSIONS + 1] = {0};
    unsigned long long local_errors = 0;
    unsigned long long local_bits = 0;
    
    // Perform multiple iterations of the entire simulation
    for (unsigned int iter = 0; iter < iterations_per_thread; iter++) {
        // Process packets
        for (int packet = 0; packet < DATA_LENGTH/HAMMING_DATA_SIZE; packet++) {
            unsigned char data[HAMMING_DATA_SIZE];
            unsigned char encoded[HAMMING_CODE_SIZE];
            unsigned char received[HAMMING_CODE_SIZE];
            unsigned char decoded[HAMMING_DATA_SIZE];
            
            // Generate random data
            for (int i = 0; i < HAMMING_DATA_SIZE; i++) {
                data[i] = curand(&localState) & 1;
            }
            
            // Simulate transmission with retransmissions
            int attempts = 0;
            bool success = false;
            
            while (!success && attempts < MAX_RETRANSMISSIONS) {
                hamming_encode(data, encoded);
                
                // Apply BSC noise
                for (int i = 0; i < HAMMING_CODE_SIZE; i++) {
                    if (curand_uniform(&localState) < bit_error_rate) {
                        received[i] = encoded[i] ^ 1;
                    } else {
                        received[i] = encoded[i];
                    }
                }
                
                success = hamming_decode(received, decoded, is_error_correcting);
                if (!success) attempts++;
            }
            
            // Count errors and update statistics
            local_retransmissions[attempts]++;
            for (int i = 0; i < HAMMING_DATA_SIZE; i++) {
                if (data[i] != decoded[i]) local_errors++;
                local_bits++;
            }
        }
    }
    
    // Atomic updates to global counters
    for (int i = 0; i <= MAX_RETRANSMISSIONS; i++) {
        atomicAdd(&results->retransmission_counts[i], local_retransmissions[i]);
    }
    atomicAdd(&results->total_errors, local_errors);
    atomicAdd(&results->total_bits, local_bits);
    
    // Save RNG state
    states[tid] = localState;
}

#include <chrono>

void run_parallel_simulation(int num_blocks, float bit_error_rate, bool is_error_correcting, unsigned int iterations_per_thread) {
    // Create CUDA events for GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    // Start CPU timer
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // Allocate device memory
    curandState* d_states;
    SimulationResult* d_results;
    
    cudaMalloc(&d_states, num_blocks * BLOCK_SIZE * sizeof(curandState));
    cudaMalloc(&d_results, sizeof(SimulationResult));
    
    // Initialize results to zero
    SimulationResult h_results = {0};
    cudaMemcpy(d_results, &h_results, sizeof(SimulationResult), cudaMemcpyHostToDevice);
    
    // Start GPU timer
    cudaEventRecord(start_gpu);
    
    // Initialize RNG states
    init_rng_kernel<<<num_blocks, BLOCK_SIZE>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    
    // Launch simulation kernel with iterations parameter
    bsc_simulation_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_states,
        d_results,
        bit_error_rate,
        is_error_correcting,
        iterations_per_thread
    );
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(&h_results, d_results, sizeof(SimulationResult), cudaMemcpyDeviceToHost);
    
    // Stop GPU timer
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    // Calculate GPU time
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start_gpu, stop_gpu);
    
    // Stop CPU timer and calculate total time
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu);
    
    // Free device memory
    cudaFree(d_states);
    cudaFree(d_results);
    
    // Clean up events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    // Print statistics
    printf("BSC Simulation Results (Iterations per thread: %u):\n", iterations_per_thread);
    printf("| Number retransmissions | number / all    | %%of time |\n");
    printf("|------------------------|-----------------|----------|\n");
    
    unsigned long long total_packets = 0;
    for (int i = 0; i <= MAX_RETRANSMISSIONS; i++) {
        total_packets += h_results.retransmission_counts[i];
    }
    
    for (int i = 0; i <= MAX_RETRANSMISSIONS; i++) {
        if (h_results.retransmission_counts[i] > 0) {
            float percentage = (h_results.retransmission_counts[i] * 100.0f) / total_packets;
            printf("| %22d | %6llu / %6llu | %7.3f%% |\n",
                   i,
                   h_results.retransmission_counts[i],
                   total_packets,
                   percentage);
        }
    }
    
    float error_rate = (float)h_results.total_errors / h_results.total_bits;
    printf("\nError rate of whole transmission: %.6f\n", error_rate);
    printf("Total bits processed: %llu\n", h_results.total_bits);
    printf("\nTiming Information:\n");
    printf("GPU Computation Time: %.3f ms\n", gpu_milliseconds);
    printf("Total Program Time:   %.3f ms\n", (float)cpu_duration.count());
    printf("Processing Speed:     %.2f Mbits/s\n", 
           (h_results.total_bits / 1000000.0) / (gpu_milliseconds / 1000.0));
}

int main(int argc, char** argv) {
    // Simulation parameters
    const int num_blocks = 32;
    const float bit_error_rate = 0.0001f;
    const bool is_error_correcting = true;
    
    // Get iterations from command line or use default
    unsigned int iterations_per_thread = 1024 * 1024;
    if (argc > 1) {
        iterations_per_thread = atoi(argv[1]);
    }
    
    // Run simulation
    run_parallel_simulation(num_blocks, bit_error_rate, is_error_correcting, iterations_per_thread);
    
    return 0;
}