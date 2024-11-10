echo "Running GCN end-to-end training benchmark..."

# python GNNA_main.py --dataset cora --dim 1433 --hidden 32 --classes 7 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode True --enable_rabbit True --loadFromTxt False

# python GNNA_main.py --dataset pubmed --dim 500 --hidden 32 --classes 3 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode True --enable_rabbit True --loadFromTxt False

# python GNNA_main.py --dataset corafull --dim 8710 --hidden 32 --classes 70 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode True --enable_rabbit True --loadFromTxt False

# python GNNA_main.py --dataset reddit --dim 602 --hidden 32 --classes 41 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode False --enable_rabbit True --loadFromTxt False

# python GNNA_main.py --dataset arxiv --dim 128 --hidden 32 --classes 40 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode False --enable_rabbit True --loadFromTxt False

# python GNNA_main.py --dataset products --dim 100 --hidden 32 --classes 47 --partSize 32 --model gcn --warpPerBlock 8 --manual_mode False --verbose_mode False --enable_rabbit True --loadFromTxt False

echo "Done"
