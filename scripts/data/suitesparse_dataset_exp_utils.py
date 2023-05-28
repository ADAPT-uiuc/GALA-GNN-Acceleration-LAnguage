import getopt as gp
import sys
import pandas as pd
import os



def parse_args(myargs):
    try:
        optlist, args = gp.getopt(myargs, 'g:n:h', ['figdir=','dir=', 'groups=', 'rmin=', 'rmax=', 'cmin=', 'cmax=', 'nzmin=', 'nzmax=', 'names=', 'degmin=', 'degmax=', 'list-groups', 'list-mtx', 'mtx-stats', 'csb-perf', 'parse', 'vec=', 'cols=', 'rmat-degs=', 'rmat-rows=', 'rmat-names=', 'help'])
    except gp.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    return optlist,args

def get_exps_to_run(optlist):
    mtx_stats = False
    csb_perf = False
    list_mtx = False
    for o, a in optlist:
        if o == "--mtx-stats":
            mtx_stats = True
        elif o == "--csb-perf":
            csb_perf = True
        elif o == "--list-mtx":
            list_mtx = True
    return mtx_stats,csb_perf,list_mtx

def vec_exps_to_run(optlist):
    mtx_stats = False
    csb_perf = False
    list_mtx = False
    for o, a in optlist:
        if o == "--vec":
            return a.split(",")
        
    return ["all"]

def get_cols_to_run(optlist):
    for o, a in optlist:
        if o == "--cols":
            return a.split(",")

def get_parse(optlist):
    for o, a in optlist:
        if o == "--parse":
            return True 
    return False            
def create_out_dir(optlist):
    for o, a in optlist:
        if o == "--dir":
            os.system("mkdir -p "+a)
            return a
def rmat_matrices(optlist):
    for o, a in optlist:
        if o == "--rmat-degs":
            degs = a.split(",")
        if o == "--rmat-rows":
            rows = a.split(",")
        if o == "--rmat-names":
            names = a.split(",")
    return rows,degs,names

def get_fig_dir(optlist):
    for o,a in optlist:
        if o=='--figdir':
            os.system("mkdir -p "+a)
            return a
        # else:
        #     os.system("mkdir -p figsdir")
        #     return a
    os.system("mkdir -p figsdir")
    return a

# 'twitter7'
ignore_list = ['3Dspectralwave', 'dielFilterV2clx', 'Chevron3', 'iChem_Jacobian', 'kim2', 'dielFilterV3clx', 'fem_hifreq_circuit', '3Dspectralwave2', 'mawi_201512020000', 'MOLIERE_2016', 'GAP-road', 'mycielskian20', 'PFlow_742', 'Chevron4', 'amazon-2008', 'tmt_sym', 'apache2', 'ca2010']

def get_matrices(optlist):
    CSV_PATH=os.path.dirname(__file__)
    data_table = pd.read_csv(CSV_PATH+'/suitesparse_datasets_sparseTamu.csv')
    data_table = data_table.loc[~data_table['Name'].isin(ignore_list)]

    for o, a in optlist:
        if o == "--nzmax":
            data_table = data_table.loc[data_table['Nonzeros'] <= int(a)] 
            # out_table = pd.concat()
        elif o == "--nzmin":
            data_table = data_table.loc[data_table['Nonzeros'] >= int(a)] 
            
        elif o == "--rmax":
            data_table = data_table.loc[data_table['Rows'] <= int(a)] 
            # out_table = pd.concat(out_table, data_table)
        elif o == "--rmin":
            data_table = data_table.loc[data_table['Rows'] >= int(a)] 
        elif o == "--degmax":
            print(a)
            lim = int(a)
            data_table = data_table.loc[data_table['Nonzeros']/data_table['Rows'] <= lim] 
        elif o == "--degmin":
            print(a)
            lim = int(a)
            data_table = data_table.loc[data_table['Nonzeros']/data_table['Rows'] >= lim] 
        elif o == "--cmax":
            data_table = data_table.loc[data_table['Cols'] <= int(a)] 
        elif o == "--cmin":
            data_table = data_table.loc[data_table['Cols'] >= int(a)] 
        elif o == "--groups":
            groups = a.split(",")
            data_table = data_table.loc[data_table['Group'].isin(groups)]
        elif o == "--names":
            names = a.split(",")
            data_table = data_table.loc[data_table['Name'].isin(names)]
    
    return data_table

def list_matrices(data_frame):
    m = data_frame.shape[0]
    return m, data_frame['Name']

def list_groups(data_frame):
    return data_frame.groupby('Group')['Name'].nunique()

def download_matrices(mtx_df, directory):
    m = mtx_df.shape[0]
    print("Downloading "+str(m)+" matrices")
    for index, row in mtx_df.iterrows():
        download_matrix(row['Group'], row['Name'], directory)
    
def download_matrix(group, name, directory, logfile=sys.stdout):
    if(os.path.exists(directory) == False):
        os.system("mkdir "+directory)
    download_str =  "https://suitesparse-collection-website.herokuapp.com/MM/"+group+"/"+name+".tar.gz"
    filename =name
    if (os.path.exists(directory+"/"+name + ".mtx") == False and os.path.exists(directory+"/"+name + ".tar.gz") == False):
        print("\tDownloading: "+str(filename), file=logfile)
        script_str = "cd "+directory+"; wget -c " + download_str + " > /dev/null 2>&1;"+"tar xzvf "+filename+".tar.gz  > /dev/null 2>&1; mv "+filename+"/"+filename+".mtx ./"
        os.system(script_str)
        
    else:
        print("\tMatrix is already donwloaded: "+str(filename), file=logfile)


def remove_matrix(group, name, directory, logfile=sys.stdout):
    filename =name
    if (os.path.exists(directory+"/"+name + ".mtx") == False and os.path.exists(directory+"/"+name + ".tar.gz") == False):
        print("\tAlready deleted "+str(filename), file=logfile)
    else:
        os.system("cd "+directory+"; rm -rf "+filename+"*")

def remove_matrix_aux_files(group, name, directory, logfile=sys.stdout):
    filename =name
    if (os.path.exists(directory+"/"+name + ".mtx") == False and os.path.exists(directory+"/"+name + ".tar.gz") == False):
        print("\tAlready deleted "+str(filename), file=logfile)
    else:
        os.system("cd "+directory+"; rm -rf "+filename+"/ "+filename+".tar.gz")