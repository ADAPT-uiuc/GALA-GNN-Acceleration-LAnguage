import argparse
import subprocess
import os

build_path = r"../../build/"

dataset_list = ["ogbn-papers100M"]

models = ["gcn"]

dgl_map = {"papers100M":"ogbn-papers100M"}

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
    if args.train:
        outfile = open(args.stat_log + "_" + args.hw + "_GALA_train.txt", 'w+')
    else:
        outfile = open(args.stat_log + "_" + args.hw + "_GALA_inf.txt", 'w+')
    outfile.write("dataset,model,hw,train,inference_time,total_time\n")
    outfile.flush()

    # TODO add build
    if not os.path.exists(build_path):
        os.makedirs(build_path)
        job_args = ['cmake',
                    '..']
        run_at(job_args, logfile, errfile, build_path)
        job_args = ['make',
                    '-j56']
        run_at(job_args, logfile, errfile, build_path)

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
            outfile.flush()
            run_at(job_args, outfile, errfile, output_path + "build/")

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def evalDGL(args):
    dgl_working_path = r"../../tests/Baselines/DGL"

    percen = [1, 2, 5, 10, 20]

    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_" + args.hw + "_DGL_node_sampling.txt", 'w+')
    outfile.write("dataset,model,hw,percen,inference_time,total_time\n")
    outfile.close()

    for pi in percen:
        for dset in dataset_list:
            for model in models:
                curr = f">>>Testing [{dset} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                print(curr)
                logfile.write(curr+"\n")
                errfile.write(curr+"\n")

                job_args = ['python',
                            '../../tests/Baselines/DGL/benchmark_dgl_'+model+'_node_sampling.py',
                            '--dataset', dgl_map[dset],
                            '--n-hidden', str(32),
                            '--pi', str(pi),
                            '--layers', str(1),
                            '--n-epochs', str(100),
                            "--logfile", args.stat_log + "_" + args.hw + "_DGL_node_sampling.txt",
                            "--device", "cuda",
                            "--skip_train",
                            "--discard", str(5)]
                outfile = open(args.stat_log + "_" + args.hw + "_DGL_node_sampling.txt", 'a+')
                outfile.write(dset + "," + model + "," + args.hw + "," + pi + ",")
                outfile.close()
                run(job_args, logfile, errfile)

                logfile.write(("<"*100)+"\n")
                errfile.write(("<"*100)+"\n")
                print("<"*100)
    logfile.close()
    errfile.close()

def evalWise(args):
    print("WiseGraph: Empty for now")

def createFigure(args):
    print("Create Figure 16-17: Empty for now")

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "dgl"):
        evalDGL(args)
    elif (args.job == "wise"):
        evalWise(args)
    elif (args.job == "fig"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info_graph_scale", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'wise', 'fig'], default="gala",
                        help="Task to generate Figures 16 to 17.")
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
