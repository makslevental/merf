python -m cProfile -o output.pstats blob.py
./gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
