python -m cProfile -o output.pstats /home/maksim/dev_projects/merf/merf/nns/dog/dog.py
./gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
