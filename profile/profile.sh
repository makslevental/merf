#PYTHONPATH=../ python -m cProfile -o output.pstats $1
PYTHONPATH=../ python -m torch.utils.bottleneck $1
#./gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
