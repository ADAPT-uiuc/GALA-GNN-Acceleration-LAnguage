# Name of the matrix to be downloaded from suite sparse
mtx_ss_name="coAuthorsCiteseer"
row_tile_size=2
column_tile_size=20000
schedule_output="../../acc_out_1/"
nPEs=5
feat_size=128
have_empty=0
reord_mtx=0

# Create a main data folder if there isn't one
if [ -d "../../data_schedule" ]
then
    echo "data_schedule folder exists"
else
    mkdir "../../data_schedule"
fi

# Create a sub data folder for the specific matrix
if [ -d "../../data_schedule/$mtx_ss_name" ]
then
    echo "data_schedule/$mtx_ss_name folder exists"
else
    mkdir "../../data_schedule/$mtx_ss_name"
fi

# Download the matrix if it doesn't exist
if [ -f "../../data_schedule/$mtx_ss_name/$mtx_ss_name.mtx" ]
then
    echo "$mtx_ss_name.mtx folder exists"
else
    python download_datasets.py --names "$mtx_ss_name" --dir "../../data_schedule/$mtx_ss_name"
fi

# Schedule the mtx_ss_name
../build/tests/accel_spmm_pretiling_test ../../data_schedule/$mtx_ss_name/$mtx_ss_name.mtx 1 100 1 100 $row_tile_size $column_tile_size $schedule_output $nPEs 1 $feat_size $have_empty $reord_mtx
