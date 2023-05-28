# declare -a MtxArray=("mycielskian12" \
# "mycielskian16" \
# "mycielskian18" \
# "luxembourg_osm" \
# "coAuthorsCiteseer" \
# "caidaRouterLevel" \
# "delaunay_n16" \
# "delaunay_n18" \
# "delaunay_n24" \
# "as-Skitter" \
# "asia_osm" \
# "coPapersCiteseer" \
# "road_usa" \
# #"kmer_V2a" \
# #"kmer_P1a" \
# "dielFilterV3real" \
# "cage15" \
# "packing-500x100x100-b050" \
# "channel-500x100x100-b050" \
# #"mawi_201512020030" \
# #"mawi_201512020130" \
# "circuit5M" \
# "wiki-topcats" \
# "mouse_gene" \
# "human_gene1" \
# "human_gene2" \
# "ldoor" \
# "com-Orkut" \
# "wikipedia-20070206" \
# "HV15R" \
# "com-LiveJournal" \
# "in-2004" \
# "venturiLevel3" \
# "Cube_Coup_dt0" \
# "Serena" \
# "ML_Laplace" \
# "soc-LiveJournal1" \
# "kron_g500-logn20" \
# "web-Google")

# declare -a MtxArray=("soc-LiveJournal1" \
# "kron_g500-logn20" \
# "web-Google")

# declare -a MtxArray=("Hook_1498" \
# "netherlands_osm" \
# "roadNet-TX")

# declare -a MtxArray=("web-Google")

# declare -a MtxArray=("me2010" \
# "fe_tooth" \
# "smallworld" \
# "fe_rotor" \
# "soc-Epinions1" \
# "soc-sign-Slashdot090221" \
# "Wordnet3" \
# "internet" \
# "usroads" \
# "delaunay_n17" \
# "id2010")

# declare -a MtxArray=("belgium_osm")

#declare -a MtxArray=("com-Friendster")
#declare -a MtxArray=("sparsine")
#declare -a MtxArray=("coAuthorsCiteseer")

#declare -a MtxArray=("mycielskian17" \
#"road_usa" \
#"asia_osm" \
#"delaunay_n24" \
#"packing-500x100x100-b050" \
#"kron_g500-logn20" \
#"coPapersCiteseer" \
#"Serena" \
#"com-LiveJournal")

#declare -a MtxArray=("pwtk" \
#"pkustk14" \
#"nd12k" \
#"af_shell1" \
#"af_shell2" \
#"af_shell3" \
#"af_shell4" \
#"af_shell5" \
#"af_shell6" \
#"af_shell7" \
#"af_shell8" \
#"af_shell9" \
#"bmw3_2" \
#"bmwcra_1" \
#"crankseg_1" \
#"crankseg_2" \
#"fcondp2" \
#"fullb" \
#"halfb" \
#"troll" \
#"Ga41As41H72" \
#"Si41Ge41H72" \
#"Si87H76" \
#"SiO2" \
#"mip1" \
#"rajat31" \
#"af_0_k101" \
#"af_1_k101" \
#"af_2_k101" \
#"af_3_k101" \
#"af_4_k101" \
#"af_5_k101" \
#"msdoor" \
#"BenElechi1" \
#"3Dspectralwave2" \
#"kkt_power" \
#"TSOPF_FS_b300_c3" \
#"atmosmodl" \
#"atmosmodm" \
#"human_gene1" \
#"human_gene2" \
#"as-Skitter" \
#"gsm_106857" \
#"delaunay_n21" \
#"hugetrace-00000" \
#"hugetric-00000" \
#"hugetric-00010" \
#"hugetric-00020" \
#"kron_g500-logn17" \
#"kron_g500-logn18" \
#"venturiLevel3" \
#"rgg_n_2_20_s0" \
#"germany_osm" \
#"great-britain_osm" \
#"italy_osm" \
#"StocF-1465" \
#"CurlCurl_3" \
#"333SP" \
#"AS365" \
#"M6" \
#"NLR" \
#"CoupCons3D" \
#"Transport" \
#"bundle_adj" \
#"mycielskian15" \
#"Hardesty1")

#declare -a MtxArray=("com-Youtube")

#declare -a MtxArray=("pwtk" \
#"pkustk14")
#declare -a MtxArray=("com-Friendster")

declare -a MtxArray=("com-Amazon" \
"belgium_osm" \
"coAuthorsCiteseer" \
"mycielskian16" \
"luxembourg_osm" )

source /home/damitha2/env_dgl/bin/activate

if [ -d "../../data_suite" ]
then
    echo "data_suite folder exists"
else
    mkdir "../../data_suite"
fi

# Read the array values with space
for val in "${MtxArray[@]}"; do
  if [ -d "../../data_suite/$val" ]
  then
      echo "data_suite/$val folder exists"
  else
      mkdir "../../data_suite/$val"
  fi
	python suitesparse_download_datasets.py --names "$val" --dir "../../data_suite/$val"
done