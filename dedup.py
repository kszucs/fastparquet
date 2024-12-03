from fastparquet import write    
import pandas as pd       
import subprocess
from PIL import Image
import os
import random
import numpy as np
import snappy
import pyarrow.parquet as pq


def generate_random_numbers(n, seed):
    # Set the seed for reproducibility
    random.seed(seed)
    
    # Generate n random numbers
    values = list(range(1, 10_000_000))
    random_numbers = random.sample(values, n)
    return random_numbers
    
    
def generate_compressible_data(n, seed, range_start, range_end, clusters):
    np.random.seed(seed)

    data = np.random.choice(np.random.randint(range_start, range_end, clusters), size=n)
    compressed_ratio = len(snappy.compress(data.tobytes())) / data.nbytes

    return data, compressed_ratio

# Example usage
nrows = 10_000_000
seed = 42
numbers, ratio = generate_compressible_data(nrows, seed, 1, 10_000_000, clusters=500_000)
print(f"Synthetic data compression ratio using snappy: {ratio:.2f}")

MB = 2**20
numbers = list(numbers)

df = pd.DataFrame({'numbers': numbers})

df_append = pd.DataFrame({'numbers': numbers + list(range(1000))})

df_update = pd.DataFrame({'numbers': numbers})
df_update.loc[int(nrows / 2)] = [0]

df_insert = pd.DataFrame({'numbers': numbers[:int(nrows/2)] + [-1] + numbers[int(nrows/2):]})
assert len(df_insert) == len(df) + 1

df_delete = pd.DataFrame({'numbers': numbers[:int(nrows/2)] + numbers[int(nrows/2)+1:]})
assert len(df_delete) == len(df) - 1


for compression in [None, "ZSTD", "LZ4", "SNAPPY"]:
    compname = "none" if compression is None else compression.lower()
    print()
    print("Fixed sized chunks")
    write(f"fsc_{compname}.parquet", df, compression=compression, page_size=1*MB)    
    write(f"fsc_{compname}_append.parquet", df_append, compression=compression, page_size=1*MB)
    write(f"fsc_{compname}_update.parquet", df_update, compression=compression, page_size=1*MB)
    write(f"fsc_{compname}_insert.parquet", df_insert, compression=compression, page_size=1*MB)
    write(f"fsc_{compname}_delete.parquet", df_delete, compression=compression, page_size=1*MB)

    print()
    print("Content defined chunks")
    write(f"cdc_{compname}.parquet", df, compression=compression, cdc=1*MB)    
    write(f"cdc_{compname}_append.parquet", df_append, compression=compression, cdc=1*MB)
    write(f"cdc_{compname}_update.parquet", df_update, compression=compression, cdc=1*MB)
    write(f"cdc_{compname}_insert.parquet", df_insert, compression=compression, cdc=1*MB)
    write(f"cdc_{compname}_delete.parquet", df_delete, compression=compression, cdc=1*MB)

    a = pq.read_table(f"fsc_{compname}.parquet")
    b = pq.read_table(f"cdc_{compname}.parquet")
    assert a.equals(b)

    a = pq.read_table(f"fsc_{compname}_append.parquet")
    b = pq.read_table(f"cdc_{compname}_append.parquet")
    assert a.equals(b)

    a = pq.read_table(f"fsc_{compname}_update.parquet")
    b = pq.read_table(f"cdc_{compname}_update.parquet")
    assert a.equals(b)

    a = pq.read_table(f"fsc_{compname}_insert.parquet")
    b = pq.read_table(f"cdc_{compname}_insert.parquet")
    assert a.equals(b)


for prefix in ["fsc", "cdc"]:
    for compname in ["none", "zstd", "lz4", "snappy"]:
        for postfix in ["append", "update", "insert", "delete"]:
            file_a = f"{prefix}_{compname}.parquet"
            file_b = f"{prefix}_{compname}_{postfix}.parquet"
            ppm_file = f"{file_b}.dedupe_image.ppm"
            jpeg_file = f"{file_b}.jpeg"
            print()
            print(f"Comparing {file_a} and {file_b}")
            subprocess.run(["./dedupe_estimator", file_a, file_b], check=True)

            with Image.open(ppm_file) as img:
                img.save(jpeg_file, "JPEG")
            print(f"Converted {ppm_file} to {jpeg_file}")
            os.remove(ppm_file)