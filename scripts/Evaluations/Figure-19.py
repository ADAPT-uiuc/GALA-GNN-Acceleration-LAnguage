import argparse
import subprocess
import os

build_path = r"../../build/"

exec_types = ["memory", "time"]
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
    outfile = open(args.stat_log + "_memory.csv", 'w+')
    outfile.write("exec,memory,total_time\n")
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

    for exec in exec_types:
        curr = f">>>Running [Execute {exec} to get time and memory consumption] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print(curr)
        logfile.write(curr+"\n")
        errfile.write(curr+"\n")

        job_args = ['../../build/tests/gala_train_memory',
                    '../../tests/GALA-DSL/ablations/memory-consumption/' + exec +'.txt',
                    output_path]
        run(job_args, logfile, errfile)

        job_args = ['make',
                    '-j56']
        run_at(job_args, logfile, errfile, output_path + "build/")
        job_args = ['./gala_model']
        outfile.write(exec + ",")
        outfile.flush()
        run_at(job_args, outfile, errfile, output_path + "build/")

        logfile.write(("<"*100)+"\n")
        errfile.write(("<"*100)+"\n")
        print("<"*100)
    logfile.close()
    errfile.close()

def evalDGL(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    job_args = ['python',
                '../../tests/Baselines/DGL/benchmark_dgl_gcn_memory.py',
                '--dataset', "RedditDataset",
                '--n-hidden', str(32),
                '--layers', str(1),
                '--n-epochs', str(100),
                "--logfile", args.stat_log + "_memory.csv",
                "--device", "cuda",
                "--discard", str(5)]
    outfile = open(args.stat_log + "_memory.csv", 'a+')
    outfile.write("dgl,")
    outfile.close()
    run(job_args, logfile, errfile)

    logfile.write(("<"*100)+"\n")
    errfile.write(("<"*100)+"\n")
    print("<"*100)
    logfile.close()
    errfile.close()

def createFigure(args):
    import pandas as pd

    time = {}
    memory = {}

    wise_df = pd.read_csv(args.stat_log + "_memory.csv")
    for index, row in wise_df.iterrows():
        print(row['exec'],'-- memory:', int(row['memory']),'-- time:',row['total_time']*1000)
        time[row['exec']] = row['total_time']*1000
        memory[row['exec']] = row['memory']

    wise_df = pd.read_csv("results_fig16_17.csv")
    for index, row in wise_df.iterrows():
        if row['hidden_feat'] == 32 and row['num_layer'] == 2 and row['model'] == 'GCN' and row['dataset'] == 'reddit':
            print('WiseGraph -- memory:',int(row['memory_used']),'-- time:',row['total_time']*1000)
            time["WiseGraph"] = row['total_time']*1000
            memory["WiseGraph"] = row['memory_used']

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    data_mem = {
        'Implementation': ['DGL', "Wise-\nGraph",  "GALA\nmemory" , "GALA\ntime"],
        'Time': [time["dgl"], time["WiseGraph"], time["memory"], time["time"]],
        'Memory': [memory["dgl"], memory["WiseGraph"], memory["memory"], memory["time"]]  # e.g., in MB
    }
    df = pd.DataFrame(data_mem)
    df['Memory'] = df['Memory'] / 1024 

    # Width for each bar (to separate them slightly)
    bar_width = 0.35
    categories = np.arange(len(df['Implementation']))

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 4))

    # Plot the 'Time' bars on the primary y-axis
    ax1.bar(categories - bar_width/2, df['Time'], width=bar_width, color='dodgerblue', label="Time(ms)", edgecolor='black')
    # ax1.set_xlabel("Category")
    ax1.set_ylabel("Runtime(ms)", color="dodgerblue")
    ax1.set_xticks(categories)
    ax1.set_xticklabels(df['Implementation'])

    # Create a second y-axis for the 'Memory' bars
    ax2 = ax1.twinx()
    ax2.bar(categories + bar_width/2, df['Memory'], width=bar_width, color='red', hatch='//', label="Memory(GB)")
    ax2.set_ylabel("Memory Used(GB)", color="red")

    plt.rcParams['hatch.linewidth'] = 3

    plt.text(2.85, 3, "1.22× faster", ha = 'center', rotation=90, fontsize=15, fontweight='bold')
    plt.text(2.2, 3.7, "2.03× less\nmemory", ha = 'center', rotation=90, fontsize=15, fontweight='bold')
    # ax1.text(2, ml, "graph", color='black', ha='left', fontsize=14)

    plt.axvline(x= 0.5, color='b', linestyle='--')
    plt.axvline(x= 1.5, color='b', linestyle='--')
    plt.axvline(x= 2.5, color='b', linestyle='--')

    # Add a combined legend
    fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize=16, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)

    plt.savefig("Figure 19.pdf", format="pdf", bbox_inches="tight", pad_inches=0, dpi=1000)

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
        evalDGL(args)
    # elif (args.job == "dgl"):
    #     evalDGL(args)
    elif (args.job == "stat"):
        createFigure(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info", help="File to store timing data")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'wise', 'stat'], default="gala",
                        help="Task to generate Figure 19.")
    parser.add_argument("--out-path", type=str,
                        default="../../", help="Output path for the generated code")
    parser.add_argument("--stdout-log", type=str,
                        default="output.log", help="File to log outputs")
    parser.add_argument("--stderr-log", type=str,
                        default="errors.log", help="File to log errors(if any)")
    args = parser.parse_args()
    main(args)
