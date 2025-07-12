import argparse
import subprocess
import os
import shutil

parser = argparse.ArgumentParser(description="Run WiseGraph test with specified hardware")
parser.add_argument("--a100", action="store_true", help="Use A100 hardware")
parser.add_argument("--h100", action="store_true", help="Use H100 hardware")
parser.add_argument("--job", type=str, choices=['F16n17', 'F19', 'F18', 'T5'], default="F16n17",
                    help="Tasks to generate the WiseGraph Numbers")
args = parser.parse_args()

if args.a100:
    hardware = "a100"
elif args.h100:
    hardware = "h100"
else:
    raise ValueError("Please specify either --a100 or --h100")

if args.job == "T5":
    script_path = os.path.abspath("../Environments/WiseGraph/" + hardware + "/CxGNN-Compute/test/ae/E1_overall/run_wisegraph-T5.sh")
    output_files = ["results_table5.csv"]
elif args.job == "F18":
    script_path = os.path.abspath("../Environments/WiseGraph/" + hardware + "/CxGNN-Compute/test/ae/E1_overall/run_wisegraph-F18.sh")
    output_files = ["results_fig18.csv"]
elif args.job == "F19":
    script_path = os.path.abspath("../Environments/WiseGraph/" + hardware + "/CxGNN-Compute/test/ae/E1_overall/run_wisegraph-F19.sh")
    output_files = ["results_fig19.csv"]
else:
    script_path = os.path.abspath("../Environments/WiseGraph/" + hardware + "/CxGNN-Compute/test/ae/E1_overall/run_wisegraph-F16-F17.sh")
    output_files = ["results_fig16_17.csv"]

working_dir = os.path.dirname(script_path)

try:
    result = subprocess.run(
        ["bash", os.path.basename(script_path), hardware],
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

for fname in output_files:
    src_path = os.path.join(working_dir, fname)
    dst_path = os.path.join(os.getcwd(), fname)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied '{fname}' to '{dst_path}'")
    else:
        print(f"Warning: '{fname}' not found in '{working_dir}'")