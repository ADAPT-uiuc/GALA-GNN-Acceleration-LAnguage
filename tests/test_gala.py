#!/usr/bin/env python3
"""Test runner for GALA DSL programs.

For each DSL file under tests/GALA-DSL/{gat,gcn,gin,sage}/:
  1. Run gala_test to generate the CUDA module
  2. Run cmake to configure the CUDA module build
  3. Run make to compile the CUDA module
  4. Run the resulting gala_model binary 5 times, recording the accuracy
     reported by each run and keeping the minimum

A test fails if any pipeline step exits with a non-zero status, or if the
minimum observed accuracy is below the threshold for that test in
tests/expected_results.json.

Results are saved to <output-dir>/results.json.

Usage:
  python tests/test_gala.py [options]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DSL_ROOT = PROJECT_ROOT / "tests" / "GALA-DSL"
ARCHS = ["gat", "gcn", "gin", "sage"]
NUM_RUNS = 5


def parse_args():
    p = argparse.ArgumentParser(description="GALA DSL test runner")
    p.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "build",
        help="Directory containing the built GALA binaries (default: <project>/build)",
    )
    p.add_argument(
        "--libtorch",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "Environments" / "libtorch" / "libtorch",
        help="Path to libtorch installation",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "Data",
        help="Root directory containing dataset subdirectories (default: <project>/Data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "output",
        help="Base directory for generated CUDA modules and build artifacts",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel jobs passed to make (default: 4)",
    )
    p.add_argument(
        "--filter",
        default=None,
        metavar="SUBSTR",
        help="Only run tests whose relative path contains SUBSTR (e.g. 'gcn/Cora')",
    )
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep generated output directories even for passing tests",
    )
    p.add_argument(
        "--expected",
        type=Path,
        default=PROJECT_ROOT / "tests" / "expected_results.json",
        help="JSON file mapping test names to minimum acceptable accuracy",
    )
    return p.parse_args()


def load_expected(path):
    """Load expected accuracy thresholds. Returns {} if the file does not exist."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def discover_tests(filter_str=None):
    tests = []
    for arch in ARCHS:
        arch_dir = DSL_ROOT / arch
        if not arch_dir.is_dir():
            continue
        for dsl_file in sorted(arch_dir.rglob("*.txt")):
            rel = dsl_file.relative_to(DSL_ROOT)
            if filter_str and filter_str not in str(rel):
                continue
            tests.append((rel, dsl_file))
    return tests


def run_step(cmd, cwd, log_fh):
    """Run cmd in cwd, writing stdout+stderr to log_fh. Returns True on success."""
    log_fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    log_fh.flush()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return result.returncode == 0


def run_model(cmd, cwd, log_fh):
    """Run cmd, writing stdout+stderr to log_fh. Returns (success, stdout_text)."""
    log_fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    log_fh.flush()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_fh.write(result.stdout)
    log_fh.write(result.stderr)
    log_fh.flush()
    return result.returncode == 0, result.stdout


def parse_accuracy(output):
    """Parse accuracy from gala_model output.

    The final output line has the form '<mean_time>,<max_acc>'.
    Returns the accuracy as a float, or None if not found.
    """
    for line in reversed(output.splitlines()):
        line = line.strip()
        if "," in line:
            try:
                return float(line.split(",")[1])
            except (ValueError, IndexError):
                continue
    return None


def run_test(rel_path, dsl_file, args):
    test_name = str(rel_path.with_suffix(""))  # e.g. "gcn/Cora/a100"
    output_dir = args.output_dir / rel_path.with_suffix("")
    log_path = args.output_dir / rel_path.with_suffix(".log")
    build_dir = output_dir / "build"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log:
        # Step 1: codegen
        if not run_step(
            [args.build_dir / "tests" / "gala_test", dsl_file, output_dir,
             "--data-root", args.data_root],
            cwd=PROJECT_ROOT,
            log_fh=log,
        ):
            return test_name, "FAIL", "codegen", log_path, None

        # Step 2: cmake
        build_dir.mkdir(exist_ok=True)
        if not run_step(
            ["cmake", output_dir,
             f"-DCMAKE_PREFIX_PATH={args.libtorch}",
             f"-DGALA_SRC_ROOT={PROJECT_ROOT}"],
            cwd=build_dir,
            log_fh=log,
        ):
            return test_name, "FAIL", "cmake", log_path, None

        # Step 3: make
        if not run_step(
            ["make", f"-j{args.jobs}"],
            cwd=build_dir,
            log_fh=log,
        ):
            return test_name, "FAIL", "make", log_path, None

        # Step 4: run NUM_RUNS times and collect accuracy
        accuracies = []
        for i in range(NUM_RUNS):
            log.write(f"\n# Run {i + 1}/{NUM_RUNS}\n")
            ok, stdout = run_model(
                [build_dir / "gala_model"],
                cwd=build_dir,
                log_fh=log,
            )
            if not ok:
                return test_name, "FAIL", f"run ({i + 1}/{NUM_RUNS})", log_path, None
            acc = parse_accuracy(stdout)
            if acc is not None:
                accuracies.append(acc)

    result_data = {
        "accuracies": accuracies,
        "min_accuracy": min(accuracies) if accuracies else None,
    }

    if not args.keep_output:
        shutil.rmtree(output_dir)

    return test_name, "PASS", None, log_path, result_data


def main():
    args = parse_args()
    tests = discover_tests(args.filter)

    if not tests:
        print("No tests found.")
        sys.exit(1)

    expected = load_expected(args.expected)
    if not expected:
        print(f"Note: no expected results loaded from {args.expected}\n")

    print(f"Running {len(tests)} tests ({NUM_RUNS} runs each)...\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    passed = failed = 0

    for rel_path, dsl_file in tests:
        test_name, status, failed_step, log_path, result_data = run_test(
            rel_path, dsl_file, args
        )
        all_results[test_name] = {"status": status}

        if status == "FAIL":
            all_results[test_name]["failed_step"] = failed_step
            print(f"  FAIL  {test_name}  [{failed_step}]  (log: {log_path})")
            failed += 1
            continue

        # Pipeline passed — record run data and check accuracy threshold.
        all_results[test_name].update(result_data)
        min_acc = result_data["min_accuracy"]
        threshold = expected.get(test_name)
        if threshold is not None:
            threshold = threshold["min_accuracy"]
            all_results[test_name]["expected_accuracy"] = threshold

        if threshold is not None and min_acc is not None and min_acc < threshold:
            all_results[test_name]["status"] = "FAIL"
            all_results[test_name]["failed_step"] = "accuracy"
            print(
                f"  FAIL  {test_name}"
                f"  [min_acc={min_acc:.2f}% < threshold={threshold:.2f}%]"
                f"  (log: {log_path})"
            )
            failed += 1
        else:
            acc_parts = []
            if min_acc is not None:
                acc_parts.append(f"min_acc={min_acc:.2f}%")
            if threshold is not None:
                acc_parts.append(f"threshold={threshold:.2f}%")
            suffix = f"  {', '.join(acc_parts)}" if acc_parts else ""
            print(f"  PASS  {test_name}{suffix}")
            passed += 1

    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    total = passed + failed
    print(f"{total} tests: {passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
