from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print(f"Number of processes: {comm.size}")

size = 10
s = np.linspace(0, 1, size)
t = np.linspace(0, 1, size)

# Get the chunk size for this correlation chunk
num_per_cpu = size // comm.size
remainder = size % comm.size

# Compute the number of columns for each CPU
sizes = num_per_cpu * np.ones(comm.size, dtype=int)
# Account for any remainder for the last rank
sizes[-1] += remainder

# Calculate the start indices of domain for this cpu
i_start = np.zeros(comm.size, dtype=int)
i_start[1:] = np.cumsum(sizes)[:-1]

# Calculate the end indices of domain for this cpu
i_end = np.cumsum(sizes)

# Make a meshgrid of the domain using the chunk needed for this CPU
ss, tt = np.meshgrid(s[i_start[comm.rank]:i_end[comm.rank]], t)
tau = ss - tt
if tau.shape[0] == tau.shape[1]:
    # If square then one off diagonal
    k = 1
else:
    # Otherwise, starts lower off diagonal since number of rows will
    # always be greater than number of columns for multiprocessing scheme
    # TODO ...unless the remainder is too big?
    k = -i_start[comm.rank] + 1

if comm.rank == 0:
    print(f"Core {comm.rank}: ({i_start[comm.rank], i_end[comm.rank]}), {tau.shape}, {k}")
    sys.stdout.flush()

verbose = True
verbose_interval = max(1, int(0.1 * num_per_cpu))

# Get the upper triangular indices for this chunk
iu, ju = np.triu_indices(n=tau.shape[0], m=tau.shape[1], k=k)
corr = np.zeros(tau.shape)

for ij in list(zip(iu, ju)):
    i, j = ij
    if verbose and (i + 1) % verbose_interval == 0 and i > 0:
        if comm.size > 1:
            print("%d%s computed on core %d" % (int(100 * (i + 1) / size), "%",
                                                             comm.rank))
            sys.stdout.flush()
        else:
            print("%d%s computed, error" % (int(100 * (i + 1) / size), "%"))
    corr[i, j] = i * tau.shape[1] + j

# Construct a list to store each of the correlation chunks
all_cpu_results = [np.array((), dtype=np.double) for _ in range(comm.size)]
all_cpu_results[comm.rank] = corr

# Compute sizes and offsets for Gatherv
sizes_memory = np.sum(sizes) * sizes
offsets = np.zeros(comm.size, dtype=int)
offsets[1:] = np.cumsum(sizes_memory)[:-1]

# Prepare buffer for Gatherv
recvbuf = np.empty(np.sum(sizes) * np.sum(sizes), dtype=np.double)
comm.Gatherv(all_cpu_results[comm.rank].flatten(),
             [recvbuf, sizes_memory.tolist(), offsets.tolist(), MPI.DOUBLE], 0)

# Let CPU 0 gather each of the correlation chunks
if comm.rank == 0:
    size = sum(sizes)
    print("Gathering parallel results...")
    ordered_chunks = []
    for i in range(comm.size):
        chunk = recvbuf[offsets[i]:offsets[i] + sizes_memory[i]].reshape(size, sizes[i])
        ordered_chunks.append(chunk)

    corr = np.hstack(ordered_chunks)
    print(f"\tNo. lags: {np.sum(sizes)}")
    print(f"\tSizes: {sizes}")
    print(f"\tMemory Size: {sizes_memory}")
    print(f"\tOffsets: {offsets}")
    print(f"Gathered correlation shape {corr.shape}")
    print(f"Correlation")

    assert (corr.shape[0] == corr.shape[1])

    il = np.tril_indices(n=corr.shape[0], m=corr.shape[0])
    corr[il] = corr.T[il]
    np.fill_diagonal(corr, 1.0)
    print(f"{corr}")
    sys.stdout.flush()
else:
    corr = None