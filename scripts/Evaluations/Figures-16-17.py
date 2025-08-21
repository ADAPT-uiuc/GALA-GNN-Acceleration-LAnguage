import argparse
import subprocess
import os

build_path = r"../../build/"

dataset_list = ["Cora",
                "Pubmed",
                "CoraFull",
                "Reddit",
                "Arxiv",
                "Products"]
models = ["gcn",
          "gat",
          "gin",
          "sage"]

dgl_map = {"Cora":"CoraGraphDataset",
           "Pubmed":"PubmedGraphDataset",
           "CoraFull":"CoraFullDataset",
           "Reddit":"RedditDataset",
           "Arxiv":"ogbn-arxiv",
           "Products":"ogbn-products"}

stir_map = {"Cora":"cora",
           "Pubmed":"pubmed",
           "CoraFull":"corafull",
           "Reddit":"reddit",
           "Arxiv":"arxiv",
           "Products":"products"}

# dataset_list = ["Cora"]
# models = ["gcn"]

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
        outfile = open(args.stat_log + "_" + args.hw + "_GALA_train.csv", 'w+')
    else:
        outfile = open(args.stat_log + "_" + args.hw + "_GALA_inf.csv", 'w+')
    outfile.write("dataset,model,hw,train,inference_time,total_time\n")
    outfile.flush()

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
                        '-DCMAKE_PREFIX_PATH="../Environments/libtorch"',
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
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_" + args.hw + "_DGL.csv", 'w+')
    outfile.write("dataset,model,hw,inference_time,total_time\n")
    outfile.close()

    for dset in dataset_list:
        for model in models:
            curr = f">>>Testing [{dset} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            job_args = ['python',
                        '../../tests/Baselines/DGL/benchmark_dgl_'+model+'.py',
                        '--dataset', dgl_map[dset],
                        '--n-hidden', str(32),
                        '--layers', str(1),
                        '--n-epochs', str(100),
                        "--logfile", args.stat_log + "_" + args.hw + "_DGL.csv",
                        "--device", "cuda",
                        "--discard", str(5)]
            outfile = open(args.stat_log + "_" + args.hw + "_DGL.csv", 'a+')
            outfile.write(dset + "," + model + "," + args.hw + ",")
            outfile.close()
            run(job_args, logfile, errfile)

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def evalSea(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_" + args.hw + "_sea.csv", 'w+')
    outfile.write("dataset,model,hw,inference_time,total_time\n")
    outfile.close()

    for dset in dataset_list:
        for model in models:
            curr = f">>>Testing [{dset} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            if (model == "gat" and args.hw == "h100"):
                outfile = open(args.stat_log + "_" + args.hw + "_sea.csv", 'a+')
                outfile.write(dset + "," + model + "," + args.hw + ",0,0\n")
                outfile.close()
            else:
                job_args = ['python',
                            '../../tests/Baselines/SeaStar/benchmark_sea_'+model+'.py',
                            '--dataset', dset,
                            '--n-hidden', str(32),
                            '--n-layers', str(1),
                            '--n-epochs', str(100),
                            "--logfile", args.stat_log + "_" + args.hw + "_sea.csv",
                            "--device", "cuda",
                            "--discard", str(5)]
                outfile = open(args.stat_log + "_" + args.hw + "_sea.csv", 'a+')
                outfile.write(dset + "," + model + "," + args.hw + ",")
                outfile.close()
                run(job_args, logfile, errfile)

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def evalSTIR(args):
    logfile = open(args.stdout_log, 'a+')
    errfile = open(args.stderr_log, 'a+')

    outfile = open(args.stat_log + "_" + args.hw + "_stir.csv", 'w+')
    outfile.write("dataset,model,hw,inference_time,total_time\n")
    outfile.close()

    for dset in dataset_list:
        for model in models:
            curr = f">>>Testing [{dset} ; ({32},{1}) ] :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print(curr)
            logfile.write(curr+"\n")
            errfile.write(curr+"\n")

            if (model == "gat"):
                outfile = open(args.stat_log + "_" + args.hw + "_stir.csv", 'a+')
                outfile.write(dset + "," + model + "," + args.hw + ",0,0\n")
                outfile.close()
            else:
                job_args = ['python',
                            '../../tests/Baselines/SparseTIR/'+model+'.py',
                            '--dataset', stir_map[dset],
                            "--logfile", args.stat_log + "_" + args.hw + "_stir.csv"]
                outfile = open(args.stat_log + "_" + args.hw + "_stir.csv", 'a+')
                outfile.write(dset + "," + model + "," + args.hw + ",")
                outfile.close()
                run(job_args, logfile, errfile)

            logfile.write(("<"*100)+"\n")
            errfile.write(("<"*100)+"\n")
            print("<"*100)
    logfile.close()
    errfile.close()

def createFigure(args):
    import pandas as pd
    import numpy as np
    import math
    import scipy
    from scipy import stats

    graph_name = {'Cora': 'Cora',
                  'Pubmed': 'Pubmed',
                  'CoraFull': 'CoraFull',
                  'Reddit': 'Reddit',
                  'Arxiv': 'OGBN-Arxiv',
                  'Products': 'OGBN-Products'}
    
    graph_name_wise = {'cora': 'Cora',
                    'pubmed': 'Pubmed',
                    'corafull': 'CoraFull',
                    'reddit': 'Reddit',
                    'ogbn-arxiv': 'Arxiv',
                    'ogbn-products': 'Products'}

    graph_name_rev = {'Cora': 'CoraGraphDataset',
                  'Pubmed':'PubmedGraphDataset',
                  'CoraFull':'CoraFullDataset',
                  'Reddit':'RedditDataset',
                  'Arxiv':'ogbn-arxiv',
                  'Products':'ogbn-products'}
    
    model_name = {
        'gat':'GAT',
        'gcn':'GCN',
        'sage':'SAGE',
        'gin':'GIN'
    }

    sname = "GALA"

    systems = ["GALA", "DGL", "SeaStar", "SparseTIR", "WiseGraph"]

    data = {}
    for mod in models:
        data[model_name[mod]] = {}
        for sys in systems:
            data[model_name[mod]][sys] = {}
            for graph in graph_name:
                data[model_name[mod]][sys][graph] = 0

    # Read GALA
    if args.train:
        gala_df = pd.read_csv(args.stat_log + "_" + args.hw + "_GALA_train.csv")
        time_field = "total_time"
    else:
        gala_df = pd.read_csv(args.stat_log + "_" + args.hw + "_GALA_inf.csv")
        time_field = "inference_time"
    for index, row in gala_df.iterrows():
        data[model_name[row['model']]]["GALA"][row['dataset']] = row[time_field]

    # Read DGL
    dgl_df = pd.read_csv(args.stat_log + "_" + args.hw + "_DGL.csv")
    for index, row in dgl_df.iterrows():
        data[model_name[row['model']]]["DGL"][row['dataset']] = row[time_field]

# dataset,model,hidden_feat,num_layer,hardware,inference_time,total_time,runtime,memory_used
    # Read WiseGraph
    wise_df = pd.read_csv("results_fig16_17.csv")
    for index, row in wise_df.iterrows():
        if row['hidden_feat'] == 32 and row['num_layer'] == 2:
            data[row['model']]["WiseGraph"][graph_name_wise[row['dataset']]] = row[time_field]

    # Read SeaStar
    dgl_df = pd.read_csv(args.stat_log + "_" + args.hw + "_sea.csv")
    for index, row in dgl_df.iterrows():
        data[model_name[row['model']]]["SeaStar"][row['dataset']] = row[time_field]

    # Read SparseTIR
    dgl_df = pd.read_csv(args.stat_log + "_" + args.hw + "_stir.csv")
    for index, row in dgl_df.iterrows():
        data[model_name[row['model']]]["SparseTIR"][row['dataset']] = row[time_field]/1000

    df_data = []
    df_cols = ["Graph", "Model", "System", "Speedup"]

    bline = sname

    skip_sys = [sname]

    max_spd = 0

    for g in graph_name:
        for mdl in data:
            for sy in data[mdl]:
                if sy in skip_sys:
                    continue
                if ((sy == 'SparseTIR')  and mdl=='GAT') or (sy == 'SeaStar' and mdl=='GAT' and args.hw == "h100"):
                    df_data.append([graph_name[g], mdl, sy, 0])
                else:
                    df_data.append([graph_name[g], mdl, sy, (data[mdl][sy][g])/(data[mdl][bline][g])])
                    if ((data[mdl][sy][g])/(data[mdl][bline][g]) > max_spd):
                        max_spd = (data[mdl][sy][g])/(data[mdl][bline][g])
    df = pd.DataFrame(df_data, columns=df_cols)

    df["x"] = df["Graph"] + "_" + df["Model"]

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(rc={'figure.figsize':(15,2)}, font_scale=1.5)

    custom_colors = [
        '#17BECF',   # Light Cyan
        '#D62728',  # Dark Red
        '#90EE90',  # Light Green
        '#B8860B',  # Dark Goldenrod
        '#D3D3D3',  # Light Gray
        '#FF7F0E',  # Dark Orange
        '#2CA02C',  # Dark Green
        '#9467BD',  # Dark Purple
        '#8C564B',  # Dark Cyan
        '#E377C2',  # Dark Magenta
        '#BCBD22',  # Dark Yellow
        '#7F7F7F',  # Dark Gray
        '#17BECF'   # Light Cyan
    ]

    if args.train:
        wld = 't'
    else:
        wld = 'i'

    if args.hw == "h100":
        dev = 1
    else:
        dev = 0

    custom_colors = custom_colors[:5 - len(skip_sys)]

    ax = sns.barplot(x='x', y='Speedup', hue='System', data=df, palette=custom_colors)

    ax.set(xticklabels=[])

    ax.set(xlabel=None)
    # ax.set(ylabel="Speedup of GALA\n(wrt baselines evaluated)")
    ax.set_ylabel("$\mathbf{GALA}$ Speedup\n(wrt baselines)", fontsize=14)
    # plt.ylabel(r"Y-axis: $\textbf{Bold Part}$ Regular Part")

    # Set scale
    plt.yscale('log', base=2)
    if (wld == 't' and dev==1):
        ticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
        plt.tick_params(axis='y', labelsize=14)
    else:
        ticks = [0.5, 1, 2, 4, 8, 16]
        # ticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
    plt.yticks(ticks, [str(tick) for tick in ticks])

    # Hatch patterns
    hatches = ['', 'o', 'x', '.', '/']
    edge_color = ['white', 'white', 'black', 'black', 'black']

    for i, bar in enumerate(ax.patches):
        if (i < 24*4):
            bar.set_hatch(hatches[i // 24])
            bar.set_edgecolor(edge_color[i // 24])
        else:
            bar.set_hatch(hatches[i % 4])
            bar.set_edgecolor(edge_color[i % 4])

    # gl = -0.13
    # ml = -0.2

    leg_anch = (0.78, 1.2)
    if wld == 'i':
        gl = 0.14
        ml = 0.21

        ax.text(-1.75, ml + 0.03, "model", color='black', ha='left', fontsize=14, rotation=0)
        ax.text(-1.75, gl, "graph", color='black', ha='left', fontsize=14)

        i = 0
        for g in graph_name:
            plt.axvline(x= i*4 + 3.5, color='b', linestyle='--')
            ax.text(1.5 + i*4, gl, graph_name[g], color='black', ha='center', fontsize=14)
            j = -1
            for mdl in data:
                ax.text(1 + i*4 + j, ml, mdl, color='black', ha='center', fontsize=12, rotation=45)
                j += 1
            i += 1

        if (dev == 1):
            ax.legend(loc='upper center', bbox_to_anchor=leg_anch, ncol=5, fontsize=14, frameon=False, borderaxespad=0.1, columnspacing=0.5, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)
            ax.text(leg_anch[0] + 11, max_spd + 5, 'Systems:', color='black', ha='center', fontsize=14)
        else:
            ax.legend().set_visible(False)
        #     ax.legend(loc='upper center', bbox_to_anchor=leg_anch, ncol=5, fontsize=14, frameon=False, borderaxespad=0.1, columnspacing=0.5, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)
        #     ax.text(leg_anch[0] + 11, max_spd + 5, 'Systems:', color='black', ha='center', fontsize=14)
    else:
        if dev == 1:
            # gl = 0.15
            # ml = 0.22
            gl = 0.07
            ml = 0.13
        else:
            gl = 0.1
            ml = 0.15

        ax.text(-1.75, ml + 0.03, "model", color='black', ha='left', fontsize=14, rotation=0)
        ax.text(-1.75, gl, "graph", color='black', ha='left', fontsize=14)

        i = 0
        for g in graph_name:
            plt.axvline(x= i*4 + 3.5, color='b', linestyle='--')
            ax.text(1.5 + i*4, gl, graph_name[g], color='black', ha='center', fontsize=14)
            j = -1
            for mdl in data:
                ax.text(1 + i*4 + j, ml, mdl, color='black', ha='center', fontsize=12, rotation=45)
                j += 1
            i += 1

        # ax.legend(loc='upper center', bbox_to_anchor=leg_anch, ncol=5, fontsize=14, frameon=False, borderaxespad=0.1, columnspacing=0.5, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)
        # ax.text(leg_anch[0] + 11, 17.5, 'Systems:', color='black', ha='center', fontsize=14)
        if dev == 1:
            ax.legend(loc='upper center', bbox_to_anchor=leg_anch, ncol=5, fontsize=14, frameon=False, borderaxespad=0.1, columnspacing=0.5, labelspacing=0.1, handletextpad=0.1, handleheight=1.1, handlelength=2)
            ax.text(leg_anch[0] + 11, max_spd + 45, 'Systems:', color='black', ha='center', fontsize=14)
        else:
            ax.legend().set_visible(False)

    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    if args.hw == "h100":
        alph = "(a)"
    else:
        alph = "(b)"

    if args.train:
        num = " 17 "
    else:
        num = " 16 "
    plt.savefig("Figure" + num + alph + ".pdf", format="pdf", bbox_inches="tight", pad_inches=0, dpi=1000)

def printStats(args):
    import pandas as pd
    import numpy as np
    import math
    import scipy
    from scipy import stats

    graph_name = {'CoraGraphDataset': 'Cora',
                  'PubmedGraphDataset': 'PubMed',
                  'CoraFullDataset': 'CoraFull',
                  'RedditDataset': 'Reddit',
                  'ogbn-arxiv': 'OGBN-Arxiv',
                  'ogbn-products': 'OGBN-Products'}

    sname = "GALA"

    data_i_h100 = {}
    data_t_h100 = {}
    data_i_a100 = {}
    data_t_a100 = {}

    # Read GALA

    # Read DGL

    # Read WiseGraph

    # Read SeaStar

    # Read SparseTIR

    df_data = []
    df_cols = ["Graph", "Model", "System", "Speedup"]

    bline = sname

    skip_sys = [sname]

    for g in graph_name:
        for mdl in data:
            for sy in data[mdl]:
                if sy in skip_sys:
                    continue
                if ((sy == 'SparseTIR')  and mdl=='GAT') or (sy == 'SeaStar' and mdl=='GAT' and dev == 1):
                    df_data.append([graph_name[g], mdl, sy, 0])
                else:
                    df_data.append([graph_name[g], mdl, sy, (data[mdl][sy][g])/(data[mdl][bline][g])])
    df = pd.DataFrame(df_data, columns=df_cols)

    df["x"] = df["Graph"] + "_" + df["Model"]

def main(args):
    if (args.job == "gala"):
        compile_and_get_time(args)
    elif (args.job == "dgl"):
        evalDGL(args)
    elif (args.job == "sea"):
        evalSea(args)
    elif (args.job == "stir"):
        evalSTIR(args)
    elif (args.job == "fig"):
        createFigure(args)
    elif (args.job == "stats"):
        printStats(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Benchmark Runner')
    parser.add_argument("--stat-log", type=str,
                        default="timing_info", help="File to store timing data")
    parser.add_argument("--hw", type=str,
                        default="h100", help="Target hardware")
    parser.add_argument("--job", type=str, choices=['gala', 'dgl', 'wise', 'sea', 'stir', 'fig', 'stats'], default="gala",
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
