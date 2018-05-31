#!/usr/bin/python

import sys
import numpy as np
import scipy.io
import utils
import networkx as nx
from sofic_processor_standalone import SoficProcessor

### Load the data file, which has the labeled Conley index rep
dat = utils.load_matlab_matrix( 'std-eps2-p2-plus34-run-dat.mat', matname='dat')

### the [0,0][0,1] retrieves the second cell entry, i.e. dat.M_inv{2}
M = dat['M_inv'][0,0][0,1]
G = utils.cell2dict(dat['G_inv'][0,0][0,1][0])

### Process the labeled Conley index information
sof = SoficProcessor(M,G,debug=True)  # Set debug to False to silence
sof.process()
print sof
print 'entropy:', sof.entropy()
### Output:
# SoficProcessor on 41 symbols, with 68 states and 127 transitions
# entropy: 0.571595590383
### Note: higher from previous work on the standard map: "Theorem 4.4. The topological entropy for the standard map for eps = 2 is bounded below by 0.54518888942276."

### Take the periodic closure and minimize
sof.take_periodic_closure(debug=True)
print "Minimizing..."
sof.minimize()
print sof
comps = list(nx.strongly_connected_components(sof.minimized_mgraph))
print "Connected components of sizes:", map(len,comps)
### Only keep one strongly connected component, in this case component 0
for c in comps[1:]:
    sof.minimized_mgraph.remove_nodes_from(c)
print sof

### The vertex shift from the original paper
SM = dat['SM'][0,0][0,1]
print "Original verified vertex shift had", SM.nnz, "edges"

### Uncomment to print the TikZ code for the graph
# print sof.sofic_shift_to_latex_graph(layout=nx.graphviz_layout)
