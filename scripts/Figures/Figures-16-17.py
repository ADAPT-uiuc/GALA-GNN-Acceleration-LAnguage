import argparse
import subprocess
import os

build_path = r"../../build/"

dataset_list = ["Cora",
                "Pubmed",
                "CoraFull",
                "Reddit",
                "ogbn-arxiv",
                "ogbn-products"]
models = ["gcn",
          "gat",
          "gin",
          "sage"]

dataset_list = ["Reddit"]
dataset_list = ["Cora"]
models = ["gcn"]

def run(args, logfile, errfile):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile)
    proc.wait()
    logfile.flush()
    errfile.flush()

def compile_and_get_time(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    # TODO add build
    # if not os.path.exists(build_path):
    #     os.makedirs(build_path)
    #     job_args = ['cmake', "../.."]
    #
    #
    # print("Exporting", args.dataset, "to", output_path)

    output_path = args.stderr_log + "codegen/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for dset in dataset_list:
        for model in models:
            curr = f">>>Running [{dset} dataset with {model} model] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['../../build/tests/gala',
                        '../../tests/GALA-DSL/' + model + '/' + dset + '/' + args.hw + '.txt',
                        output_path]

            run(job_args, logfile, errfile)

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def main(args):
    compile_and_get_time(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info.csv", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
