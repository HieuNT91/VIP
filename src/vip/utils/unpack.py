
# touch ./data_folder/transfer_{0..27}.txt
# python unpack.py ./data_folder -o my_model.json

import base64
import zlib
import glob
import os
import re
import argparse

def unpack(input_folder, output_file):
    # 1. Find chunks
    search_path = os.path.join(input_folder, "transfer_*.txt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Error: No 'transfer_*.txt' files found in '{input_folder}'")
        return

    # 2. Sort numerically
    files.sort(key=lambda f: int(re.search(r'transfer_(\d+).txt', f).group(1)))

    print(f"Found {len(files)} chunks. Merging...")

    # 3. Read and Merge
    full_b64_parts = []
    for fname in files:
        with open(fname, "r") as f:
            full_b64_parts.append(f.read().strip())
    
    full_b64_string = "".join(full_b64_parts)

    try:
        # 4. Decode & Decompress
        print("Decoding Base64...")
        compressed_data = base64.b64decode(full_b64_string)
        
        print("Decompressing...")
        original_data = zlib.decompress(compressed_data)
        
        # 5. Save
        with open(output_file, "wb") as f:
            f.write(original_data)
            
        print(f"Success! Restored to '{output_file}' ({len(original_data)/1024/1024:.2f} MB)")
        
    except Exception as e:
        print(f"FAILED. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack text chunks back into original file.")
    parser.add_argument("input_folder", type=str, help="Folder containing the transfer_*.txt files")
    parser.add_argument("-o", "--output", type=str, default="restored_data.json", help="Output filename (default: restored_data.json)")
    
    args = parser.parse_args()
    unpack(args.input_folder, args.output)