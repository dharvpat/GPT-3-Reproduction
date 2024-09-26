import os

def shard_data(data, shard_size):
    # Split data into smaller shards for efficient processing
    shards = [data[i:i + shard_size] for i in range(0, len(data), shard_size)]
    return shards

def load_shards(dataset_dir, shard_size_gb):
    # Determine the approximate number of shards based on file sizes
    shard_size_bytes = shard_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    shard_paths = []
    
    # Simulating shard calculation (would be based on real dataset size)
    total_dataset_size = sum(os.path.getsize(os.path.join(dataset_dir, f)) for f in os.listdir(dataset_dir))
    num_shards = max(1, total_dataset_size // shard_size_bytes)
    
    # Split the files into shards (in reality, more sophisticated logic may be needed)
    files = sorted(os.listdir(dataset_dir))
    for i in range(num_shards):
        shard_files = files[i::num_shards]  # Load a subset of files for each shard
        shard_paths.append([os.path.join(dataset_dir, file) for file in shard_files])
    
    return shard_paths

def save_data(data, output_dir, filename):
    # Save data to the specified directory and filename
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        for line in data:
            f.write(f"{line}\n")