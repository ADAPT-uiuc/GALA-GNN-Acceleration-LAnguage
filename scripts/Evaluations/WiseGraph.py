import subprocess
import os
import shutil

# Path to the bash script
script_path = os.path.abspath("../../../CxGNN-Compute/test/ae/E1_overall/run_wisegraph.sh")
working_dir = os.path.dirname(script_path)

# Run the bash script in its own directory
try:
    result = subprocess.run(
        ["bash", os.path.basename(script_path)],
        cwd=working_dir,
        check=True,
        capture_output=True,
        text=True
    )
    print("Script executed successfully.")
    print("Output:\n", result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)
except subprocess.CalledProcessError as e:
    print("Script failed with return code:", e.returncode)
    print("Error output:\n", e.stderr)

# Copy result CSVs to the current directory
output_files = ["results_fig16_17.csv", "results_fig18_19.csv", "results_table5.csv"]
for fname in output_files:
    src_path = os.path.join(working_dir, fname)
    dst_path = os.path.join(os.getcwd(), fname)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied '{fname}' to '{dst_path}'")
    else:
        print(f"Warning: '{fname}' not found in '{working_dir}'")