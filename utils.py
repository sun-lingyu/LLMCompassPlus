import subprocess
import re
import os
import glob
import sys
from typing import List

def size_of_list(list: List):
    result = 1
    for i in list:
        result *= i
    return result

def size(list):
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return len(list)

def closest_factors(n):
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0

def run_command(target_args):
    # Define the nsys command
    output_base = "temp_profile_report"
    
    # Check if script_name is provided
    if not target_args:
        print("Error: No script or command provided to profile.")
        print("Usage: python run_profile.py <script.py> [args]")
        sys.exit(1)

    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--capture-range=cudaProfilerApi",
        "--stop-on-exit=true",
        "-o", output_base,
        "-f", "true",
    ] + target_args

    print(f"Running command: {' '.join(cmd)}")
    
    # We capture output. nsys (v2023/2024) writes stats to stdout.
    # Note: nsys might return non-zero exit code (e.g. 143 SIGTERM) even on success if stopped via stop-on-exit?
    # We will ignore return code if we get valid stats.
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    full_output = result.stdout + "\n" + result.stderr
    return full_output, output_base

def parse_nsys_stats(output):
    if not output:
        return

    # We look for the section "cuda_gpu_kern_sum"
    # And then the header line
    
    header_regex = re.compile(r"Time \(%\)\s+Total Time \((.+?)\).*?Min \((.+?)\).*?Name")
    
    lines = output.splitlines()
    
    current_section = None
    kernels_min_times = []
    
    min_time_col_index = 5 # Default, verify with header
    
    for i, line in enumerate(lines):
        if "stats report" in line:
            if "cuda_gpu_kern_sum" in line:
                current_section = "kernels"
            else:
                current_section = "other"
            continue
            
        if current_section == "kernels":
            match = header_regex.search(line)
            if match:
                # Found header in kernel section
                unit = match.group(2) # Group 2 is Min unit ('ns')
                continue
            
            if "------" in line:
                continue
                
            if not line.strip():
                if kernels_min_times:
                    # End of table
                    current_section = None
                continue

            # distinct header check to avoid parsing header as data
            if "Total Time" in line: 
                continue

            # Data line
            parts = line.split()
            if len(parts) > min_time_col_index:
                try:
                    # nsys uses comma separators for thousands
                    val_str = parts[min_time_col_index].replace(',', '')
                    val = float(val_str)
                    
                    kernels_min_times.append(val)
                except ValueError:
                    pass
                
    if kernels_min_times:
        return kernels_min_times
    else:
        return None

def cleanup_files(output_base):
    print("\nCleaning up report files...")
    patterns = [f"{output_base}*.nsys-rep", f"{output_base}*.sqlite"]
    for p in patterns:
        for f in glob.glob(p):
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
