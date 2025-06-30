import os
import re
import csv
from statistics import mean

def parse_log_file(filepath, hardware='h100', output_csv="parsed_results.csv"):
    with open(filepath, 'r') as f:
        text = f.read()

    sections = re.split(r"Namespace\(", text)
    results = []

    for sec in sections[1:]:
        ns_block = "Namespace(" + sec.split("Benchmark model inference", 1)[0]
        ns_fields = ns_block.split(',')
        dataset = model = hidden_feat = num_layer = None
        for field in ns_fields:
            if 'dataset=' in field:
                dataset = field.split('=')[1].strip().strip("'\"")
            elif 'model=' in field:
                model = field.split('=')[1].strip().strip("'\"")
            elif 'hidden_feat=' in field:
                hidden_feat = field.split('=')[1].strip().strip("'\"")
            elif 'num_layer=' in field:
                num_layer = field.split('=')[1].strip().strip("'\"")

        section_text = "Namespace(" + sec
        inf_times = re.findall(r"Inference Time:\s*([0-9.]+)", section_text)
        inf_times = list(map(float, inf_times)) if inf_times else []

        total_times = re.findall(r"Training Time\s*([0-9.]+)", section_text)
        total_times = list(map(float, total_times)) if total_times else []

        if dataset and model and inf_times:
            runtime_match = re.search(rf"CSR_Layer\s+{model}\s*:\s*([0-9.]+)", section_text)
            memory_match = re.search(r"Average GPU Used/Total.*?:\s*([0-9.]+)/", section_text)
            results.append({
                "dataset": dataset,
                "model": model,
                "hidden_feat": hidden_feat,
                "num_layer": num_layer,
                "hardware": hardware,
                "inference_time": round(mean(inf_times), 8) if inf_times else None,
                "total_time": round(mean(total_times), 8) if total_times else None,
                "runtime": float(runtime_match.group(1)) if runtime_match else None,
                "memory_used": float(memory_match.group(1)) if memory_match else None
            })

    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {output_csv}")
    else:
        print("No valid entries parsed.")

    return results

def parse_logs_from_folder(folder_path, hardware='h100', output_csv="parsed_results.csv"):
    results = []
    for file in os.listdir(folder_path):
        if file.endswith(".log") or file.endswith(".txt"):
            parsed = parse_log_file(os.path.join(folder_path, file), hardware)
            if parsed:
                results.extend(parsed)

    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} entries to {output_csv}")
    else:
        print("No valid entries parsed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="results_fig16_17.log", help="Path to a single log file")
    parser.add_argument("--folder", type=str, help="Path to folder with log files")
    parser.add_argument("--hardware", type=str, default="h100", help="Hardware label")
    parser.add_argument("--output", type=str, default="results_fig16_17.csv", help="Output CSV file name")
    args = parser.parse_args()

    if args.log_file:
        parse_log_file(args.log_file, hardware=args.hardware, output_csv=args.output)
    elif args.folder:
        parse_logs_from_folder(args.folder, hardware=args.hardware, output_csv=args.output)
    else:
        print("Please specify either --log_file or --folder.")