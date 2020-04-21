PYTHONPATH=../ python -m cProfile -o output.pstats $1
# FSP_DATA_DIR=~/data/FSP_DATA ./profile.sh ../nns/dog/dog.py > cpu_profile
#PYTHONPATH=../ python -m torch.utils.bottleneck $1
#./gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
