#!/usr/bin/env python3

from analyze_embeddings import *
from analyze_metric import *

"""
Top-level scripts that produces results for all runs and writes them to file
Calculates metric statistics across multiple runs

"""

log_file = open('log_file.txt','w')

for run in range(2,6):
    log_file.write('### Analysis for Run {} ### \n\n'.format(run))
    log_file.write('*** Embeddings Analysis ***\n')
    analyze_embeddings(run=str(run),draw=False,log_file=log_file)
    log_file.write('\n')
    log_file.write('*** Metric Analysis ***\n')
    analyze_metric(run=str(run),draw=False,log_file=log_file)
    log_file.write('\n\n\n')


log_file.close()



