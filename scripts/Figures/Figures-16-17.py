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
models = ["gcn"]

def run(args, logfile, errfile):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile)
    proc.wait()
    logfile.flush()
    errfile.flush()

def run_at(args, logfile, errfile, path):
    proc = subprocess.Popen(args, stdout=logfile, stderr=errfile, cwd=path)
    proc.wait()
    logfile.flush()
    errfile.flush()

def compile_and_get_time(args):
    logfile = open(args.stdout_log, 'w+')
    errfile = open(args.stderr_log, 'w+')
    outfile = open(args.stat_log, 'w+')
    outfile.write("dataset,model,hw,train,inference_time,total_time\n")

    # TODO add build
    # if not os.path.exists(build_path):
    #     os.makedirs(build_path)
    #     job_args = ['cmake', "../.."]
    #
    #
    # print("Exporting", args.dataset, "to", output_path)

    output_path = args.out_path + "codegen/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_path + "build/"):
        os.makedirs(output_path + "build/")

    for dset in dataset_list:
        for model in models:
            curr = f">>>Running [{dset} dataset with {model} model] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            if args.train:
                job_args = ['../../build/tests/gala_train',
                            '../../tests/GALA-DSL/' + model + '/' + dset + '/' + args.hw + '.txt',
                            output_path]
            else:
                job_args = ['../../build/tests/gala_inference',
                            '../../tests/GALA-DSL/' + model + '/' + dset + '/' + args.hw + '.txt',
                            output_path]
            run(job_args, logfile, errfile)

            curr = f">>>Building the code for [{dset} dataset with {model} model] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['cmake',
                        '-DCMAKE_PREFIX_PATH="/home/damitha2/new_torch/libtorch"',
                        '-DCAFFE2_USE_CUDNN=True',
                        '..']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['make',
                        '-j56']
            run_at(job_args, logfile, errfile, output_path + "build/")
            job_args = ['./gala_model']
            intTrain = int(args.train == 'true')
            outfile.write(dset + "," + model + "," + args.hw + "," + str(intTrain) + ",")
            run_at(job_args, outfile, errfile, output_path + "build/")

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
                        default="timing_info.txt", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--train", action='store_true',
                        help="Train the model")
    parser.set_defaults(train=False)
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
