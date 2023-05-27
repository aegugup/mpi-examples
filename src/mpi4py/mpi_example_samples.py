from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print(f"Number of processes: {comm.size}")


num_domain = 3
size = 20

# Calculate number of samples for each CPU
num_per_cpu = size // comm.size

# Assign the number of rows for each CPU
sizes = num_per_cpu * np.ones(comm.size, dtype=int)
# Account for any remainder for the last rank
sizes[-1] += size % comm.size
# Construct some data
samples = np.arange(sizes[comm.rank] * num_domain, dtype=np.double).reshape(sizes[comm.rank], num_domain)

# Construct a list to store each of the sample chunks
all_cpu_results = [np.array((), dtype=np.double) for _ in range(comm.size)]
all_cpu_results[comm.rank] = samples

# Compute sizes and offsets for Gatherv
sizes_memory = num_domain * sizes
offsets = np.zeros(comm.size, dtype=int)
offsets[1:] = np.cumsum(sizes_memory)[:-1]

# Prepare buffer for Gatherv
recvbuf = np.empty(np.sum(sizes) * num_domain, dtype=np.double)
comm.Gatherv(all_cpu_results[comm.rank].flatten(),
             [recvbuf, sizes_memory.tolist(), offsets.tolist(), MPI.DOUBLE], 0)

# Let CPU 0 gather each of the sample chunks
if comm.rank == 0:
    samples = recvbuf.reshape((np.sum(sizes), num_domain))
    print(f"\tNo. samples: {np.sum(sizes)}")
    print(f"\tSizes: {sizes}")
    print(f"\tMemory Size: {sizes_memory}")
    print(f"\tOffsets: {offsets}")
    print(f"\tGathered samples shape {samples.shape}")
    print(f"Samples")
    print(f"{samples}")
    sys.stdout.flush()
else:
    samples = None
