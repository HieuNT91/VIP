# python pack.py data.json --size 2

import base64
import zlib
import os
import argparse

def pack_and_split(input_file, chunk_size_mb):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # Calculate bytes
    chunk_size_bytes = int(chunk_size_mb * 1_000_000)

    # Determine output folder name based on input filename
    # e.g., /path/to/data.json -> /path/to/data
    folder_name = os.path.splitext(input_file)[0]
    
    os.makedirs(folder_name, exist_ok=True)
    print(f"Output folder: {folder_name}/")

    # 1. Read
    with open(input_file, 'rb') as f:
        raw_data = f.read()
    print(f"Original Size: {len(raw_data)/1024/1024:.2f} MB")

    # 2. Compress
    compressed_data = zlib.compress(raw_data, level=9)
    
    # 3. Encode
    b64_data = base64.b64encode(compressed_data).decode('utf-8')
    
    total_len = len(b64_data)
    print(f"Compressed & Encoded Size: {total_len/1024/1024:.2f} MB")
    
    # 4. Split
    num_chunks = (total_len + chunk_size_bytes - 1) // chunk_size_bytes
    print(f"Splitting into {num_chunks} chunk(s) of {chunk_size_mb}MB...")

    for i in range(num_chunks):
        chunk = b64_data[i*chunk_size_bytes : (i+1)*chunk_size_bytes]
        
        output_path = os.path.join(folder_name, f"transfer_{i}.txt")
        with open(output_path, "w") as out:
            out.write(chunk)
        print(f"-> Created {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack and split a file for text transfer.")
    parser.add_argument("input_file", type=str, help="Path to the file to pack")
    parser.add_argument("--size", type=float, default=1.0, help="Chunk size in MB (default: 1.0)")
    
    args = parser.parse_args()
    pack_and_split(args.input_file, args.size)