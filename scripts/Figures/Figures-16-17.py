import argparse
import subprocess
import os

build_path = r"../../build/"

dataset_list = ["CoraGraphDataset",
                "PubmedGraphDataset",
                "CoraFullDataset",
                "RedditDataset",
                "ogbn-arxiv",
                "ogbn-products"]
models = ["gcn",
          "gat",
          "gin",
          "sage"]

dataset_list = ["CoraGraphDataset"]
models = ["gcn"]

def run(args, logfile, errfile):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile)
    proc.wait()
    logfile.flush()
    errfile.flush()

def get_npy(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    # TODO add build
    # if not os.path.exists(build_path):
    #     os.makedirs(build_path)
    #     job_args = ['cmake', "../.."]
    #
    #
    # print("Exporting", args.dataset, "to", output_path)

    for dset in dataset_list:
        for model in models:
            curr = f">>>Exporting [{dset}] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['../../build/gala',
                        dset, model, args.hw]

            run(job_args, logfile, errfile)

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def main(args):
    get_npy(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info.csv", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
