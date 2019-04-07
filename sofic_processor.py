"""
sofic_processor.py

Opened: December 2012

Author: Rafael Frongillo

Processes a labeled Conley index representative by finding a sofic
shift which is semiconjugate to the original system.  Requires NetworkX 1.11 (and does not work with 2.*).
"""

# technically speaking we should wrap this; using only for
# MultiDiGraph currently (the correct data structure for edge
# presentations of sofic shifts...)
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout

from collections import deque # for processing loop; queue okay in retrospect
import numpy as np
import sympy as sp
from fractions import gcd # for column echelon form
import string

from DFA import DFA # for minimization


############################################################
# GENERAL ALGORITHMS
############################################################

def log_max_eigenvalue(S):
    """
    Returns max(log( | v | )), where v is the leading
    eigenvalue of M.

    S -- Transition graph or numpy matrix
    """
    # S is not an array, so it must be a DiGraph object (no type
    # checking past this).
    if isinstance(S, DiGraph):
        M = nx.to_numpy_matrix(S)
    # S is already a numpy matrix
    elif isinstance(S, (np.matrix, np.ndarray)):
        M = S
    # compute the eigenvalues
    v = np.linalg.eigvals(M)
    # log of the maximum modulus
    return np.log(np.absolute(v).max())

def condensation( G, components, loops=False ):
    """Return the condensation of G with respect to the given components.

    Given components C_1 .. C_k which partition the nodes of G, the
    condensation graph cG has nodes <C_1> .. <C_k> and has edge
    (<C_i>,<C_j>) iff there was an edge in G from a node in C_i to a
    node in C_j.
    
    Note: the given components can be arbitrary, but if they are the
    strongly connected components (in the graph theory sense, not the
    dynamics sense) of G, then cG corresponds to the directed acyclic
    graph between strongly connected components.

    Parameters
    ----------
    G : nx.DiGraph

    components : list of lists of component nodes, or dictionary of
    component label -> component nodes

    loops : whether to allow condensed nodes to have self loops (default: False)

    Returns
    -------
    cG : nx.DiGraph
       The condensed graph.
    """
    # convert list to dict
    if isinstance(components,list):
        components = {c:components[c] for c in range(len(components))}
        
    mapping = {n:c for c in components for n in components[c]}
    cG = nx.DiGraph()
    for u in mapping.keys():
        cG.add_node(mapping[u])
        for v in G.successors(u):
            # if v~u and u,v are in the same component, don't add the loop
            # (unless we allow loops)
            if loops or v not in components[mapping[u]]:
                cG.add_edge(mapping[u], mapping[v])
    return cG

def condensation_multi( G, components ):
    """Return the multigraph given by the condensation of G with
    respect to the given components.

    NOTE: preserves data on the edges, but currently only works when
    the data has only one key.
    
    Given components C_1 .. C_k which partition the nodes of G, the
    condensation graph cG has nodes <C_1> .. <C_k> and has edge
    (<C_i>,<C_j>) iff there was an edge in G from a node in C_i to a
    node in C_j.
    
    Parameters
    ----------
    G : nx.DiGraph or nx.MultiDiGraph

    components : list of lists of component nodes, or dictionary of
    component label -> list of component nodes

    Returns
    -------
    cG : nx.DiGraph
       The condensed graph.
    """
    # convert lists to tuples for hashing, list to dict
    if isinstance(components,dict):
        components = {c:tuple(components[c]) for c in components.keys()}
    elif isinstance(components,list):
        components = {c:tuple(components[c]) for c in range(len(components))}

    mapping = {n:components[c] for c in components for n in components[c]}
    cG = nx.MultiDiGraph()
    cG.add_nodes_from(components.values())
    for (u,v,a) in G.edges_iter(data=True):
        (key,val) = a.keys()[0],a.values()[0]
        dat = cG.get_edge_data(mapping[u], mapping[v])
        # only add an edge (u,v,key=val) if there isn't already a (u,v) edge with key=val
        if not dat or val not in [d[key] for d in dat.values()]:
            cG.add_edge(mapping[u], mapping[v], **a)
    return cG




############################################################
# HashableMatrix
############################################################

def lcm(a,b):
    return a*b // gcd(a,b)

class HashableMatrix(object):
    """
    A wrapper around a numpy matrix with a __hash__ function.
    """
    def __init__(self, mat):
        # in case we are given a sparse matrix, cast it
        # NOTE: not currently handling all cases...
        if hasattr(mat, 'todense'):
            self.mat = mat.todense()
        elif type(mat) != np.matrix:
            self.mat = np.matrix(mat)     #pray that this works
        else:
            self.mat = mat

    def __hash__(self):
        return hash(str(self.mat))

    def __repr__(self):
        s = str(self.mat.astype(np.int).tolist())
        s = string.replace(s,",","")
        s = string.replace(s,"[[","[")
        s = string.replace(s,"] [","; ")
        s = string.replace(s,"]]","]")
        return "H" + s
        #       return "H" + string.replace(str(self.mat),"\n",";")

    def __eq__(self, other):
        if isinstance(other, HashableMatrix):
            return np.all(self.mat == other.mat)
        else:
            print self, other
            return False
        raise NotImplemented

    def iszero(self):
        """
        Returns True iff this is the zero matrix (or the empty matrix,
        which is the normalization of the zero matrix)
        """
        return not np.any(self.mat)
    
    def normalized_gcd(self):
        """
        Returns a HashableMatrix given by dividing this matrix by the greatest
        common divisor of the nonzero entries and normalizes the sign by
        the first nonzero entry.  If this is the zero matrix, this
        matrix is returned.
        """
        if self.iszero(): # Already the zero matrix!
            return HashableMatrix(self.mat.copy())
        # list of nonzero entries
        nz = self.mat[np.nonzero(self.mat)].tolist()[0]
        # sign of 1st nz * gcd of nz's
        return HashableMatrix( self.mat * np.sign(nz[0])
                               / reduce(gcd, map(abs, nz)) )

    def normalized(self):
        """
        Returns a HashableMatrix describing the image of this matrix.
        """
        # swap cols and rows and row reduce
        A = sp.Matrix(self.mat).transpose().rref()[0]
        # ensure all entries are integral by multiplying by the LCD
        A *= reduce(lcm, map(lambda x: x.as_numer_denom()[1], # get denominators
                             sp.utilities.iterables.flatten(A)))
        # see https://groups.google.com/forum/#!topic/sympy/e8hcF4QAldc
        A = np.array(sp.lambdify((), A, 'numpy')()) # convert to numpy array
        A = A[np.any(A,1),:].transpose()    # remove 0 rows, swap rows and columns
        return HashableMatrix(np.matrix(A))
    
        

    
############################################################
# SoficProcessor
############################################################

class SoficProcessor(object):
    """
    Processes a given index map of generators into a sofic shift.

    NOTE ON EDGE DIRECTION: to match the interpretation of the index
    map as a linear transformation, we consider index_map[i,j] to be
    the coefficient of generator i in the image of generator j.  This
    means our matrix multiplication along a path of symbols (i,j,k)
    will index_map[g(k),g(j)] * index_map[g(j),g(i)], where g(i) is
    the indices of generators corresponding to symbol i.  We will
    however continue to write (i,j) and (j,k) in the *graph*, as is
    standard.
    
    Pseudocode for the algorithm, in slightly different notation:

    INPUT: graph G on symbols with edge (s,t) labeled by M(s,t), the
      index map matrix from region s to region t
    OUTPUT: a graph H which is the right-resolving representation of
      a sofic shift semi-conjugate to the original system
     
    let H be a graph with nodes {[t,ech(M(s,t))] for all edges (s,t) of G}
      where ech(M) is the column echelon form of M
    let Q be a queue initially containing all nodes of H
     
    loop until Q nonempty:
      dequeue some node [t,M] from Q
      for edges (t,u) in G:
        let M' = ech(M(t,u) * M)
        if M' is not the zero matrix:
          if [u,M'] is not a node of H, add to H and Q
          add edge ([t,M], [u,M']) labeled 'u' to H
 
    (remove nodes of H not in the graph maximal invariant set)
    """
    def __init__(self,
                  index_map=None,
                  reg2gen=None,
        #symbol_transitions=None,
                  debug=False):
        """
        index_map -- numpy matrix of map on generators on disjoint
        regions of an attactor in phase space. Output from CHomP or
        other homology software.

        reg2gen -- a dictionary of lists representing generators
        per region.

        debug -- turn on debugging messages (default = False)
        """

        self.debug = debug

        # NOTE: when index_map is sparse, it breaks all sorts of
        # things right now.  TODO: implement the sparse case (or just
        # forget about the overhead, since we don't see too many
        # examples with 10k generators...)
        if hasattr(index_map,'todense'):
            index_map = index_map.todense()
            if self.debug:
                print "Made index_map dense; new type =",type(index_map)
        
        # construct the starting graph on symbols
        # by the above, we must reverse the edge direction!
        generator_graph = DiGraph(index_map.transpose())
        self.symbol_graph = condensation(generator_graph,
                                         reg2gen,
                                         loops=True)

        # dictionary of matrices from the index map for each symbol
        # transition in symbol_graph
        self.edge_matrices = {}
        
        # graph with nodes (symbol,matrix)
        self.mgraph = DiGraph()
        for e in self.symbol_graph.edges_iter():
            # hashable, as we will use it in dictionaries.
            # NOTE: an edge (i,j) means we are mapping generators in
            # region i to those in region j, so this is a n_j by n_i
            # matrix
            mat = HashableMatrix( index_map[np.ix_(reg2gen[e[1]],
                                                   reg2gen[e[0]])] )
            self.edge_matrices[e] = mat
            # we are mapping to region j, with matrix mat
            self.mgraph.add_node( (e[1],mat.normalized()) )

        # nodes to explore next; init with all (symbol,matrix) nodes
        self.explore_nodes = deque(self.mgraph.nodes())

        # add the special nodes
        # for s in self.symbol_graph.nodes():
        #     self.mgraph.add_node((s,'zero'))
        # self.mgraph.add_node('unexplored')

        # instead of an explicit 'zero' node, keep a list of forbidden
        # transitions, of the form (node, symbol)
        self.forbidden_transitions = []

        # track how many processing steps total:
        self.processing_steps = 0
        
    def matrix_product(self, node, symbol):
        """
        Compute the normalized matrix product of (s,M) and t
        """
        # NOTE the order of multiplication: M is a matrix mapping into
        # region s, and E=edge_matrices[(s,t)] is mapping from s to t,
        # so we want S*M
        return HashableMatrix(self.edge_matrices[(node[0],symbol)].mat
                               * node[1].mat
                              ).normalized()

                                
        
    def process(self, steps=np.Inf, debug=None):
        """
        Process the index information for some number of steps.
        Stores all the nodes to explore in self.explore_nodes, so that
        this method may be run again to pick up where it left off.  See
        has_terminated()

        steps -- how many steps to run (default = infinity)
        debug -- whether to print diagnostic info (default = None)
        """
        if debug == None:
            debug = self.debug

        # hack to figure out "real" step count
        # stepcount = len(self.explore_nodes)
        
        while self.explore_nodes and steps > 0:
            node = self.explore_nodes.popleft()   # breadth-first
            if debug:
                print self
                print "Exploring node", node
            s = node[0]                         # current vertex
            steps -= 1                          # decrement step count
            self.processing_steps += 1          # increment total steps

            # for all t out of s
            for t in self.symbol_graph.successors(s):
                if debug:
                    print "Transition:", (s,t), self.edge_matrices[(s,t)]

                # hashable matrix which is the product up to t
                matrix = self.matrix_product(node,t)

                if matrix.iszero():
                    self.forbidden_transitions.append( (node,t) )
                    if debug:
                        print "FORBIDDEN!  ", node, "->", (t,0)

                else:
                    new_node = (t,matrix)
                    if new_node not in self.mgraph:
                        if debug:
                            print "Adding new node:", new_node
                        self.explore_nodes.append(new_node)
                    self.mgraph.add_edge(node,new_node,label=t)
                    if debug:
                        print "Adding edge:", node, "->", new_node
            if debug:
                print "Have completed", self.processing_steps, "total steps"

        if debug:
            if self.has_terminated():
                print "Finished processing"
            else:
                print "Has not finished processing"
            print self

                        
    def entropy(self):
        """
        Return the topological entropy of the sofic shift up to the
        current level of processing.
        """
        return log_max_eigenvalue(nx.to_numpy_matrix(self.mgraph))

    def num_symbols(self):
        """
        Return how many symbols the sofic processor is using.
        """
        return self.symbol_graph.number_of_nodes()

    def symbols(self):
        """
        Return the symbol set used.
        """
        return range(self.num_symbols())

    def has_terminated(self):
        """
        Return a boolean which is True if and only if the processor
        has completed processing, i.e., no marked nodes remain.
        """
        return len(self.explore_nodes)==0

    def take_periodic_closure(self,debug=None):
        """
        Removes edges between distinct strongly-connected components
        of the underlying vertex presentation, and then removes any
        nodes with no edges.  Should be called before minimize().
        """
        comps = list(nx.strongly_connected_components(self.mgraph))
        node2comp = {n:c for c in range(len(comps)) for n in comps[c]}

        if debug:
            print "Connected components of sizes:", map(len,comps)
        
        for e in self.mgraph.edges():
            if not node2comp[e[0]] == node2comp[e[1]]:
                # print 'edge between components:', e
                self.mgraph.remove_edge(*e)

        for n in self.mgraph.nodes():
            # no outgoing edges means no incoming edges either
            if not self.mgraph.successors(n):
                self.mgraph.remove_node(n)

                
    def to_DFA(self):
        """
        Returns a Deterministic Finite Automaton (DFA) representation
        of the sofic shift, solely for the purpose of running DFA
        minimization algorithms.  The DFA is encoded in a numpy array
        where A[state,symbol] = nextstate.  Here (#states - 1) is the
        reject state, and there is no initial state.
        """
        states = self.mgraph.nodes() + ['reject']
        reject = len(states)-1 # reject state index
        # accepts = set(range(len(states)))-set([reject]) # all but reject

        # transitions(node,symbol) = symbol
        # default transition is reject; this has the correct behavior
        # as missing transitions should point to reject, and reject
        # followed by any symbol gives reject
        transitions = np.full( (len(states),self.num_symbols()),
                               reject, dtype=np.int )

        # mapping nodes to DFA states
        nodehash = {n:states.index(n) for n in states}

        for s,t,data in self.mgraph.edges_iter(data=True):
            transitions[nodehash[s],data['label']] = nodehash[t]

        if self.debug:
            print 'DFA transitions:'
            print transitions
            
        return DFA(states=states,
                   num_symbols=self.num_symbols(),
                   transitions=transitions,
                   reject=reject)


    def minimize(self):
        """
        Stores to self.minimized_mgraph the edge presentation of the
        sofic shift presented by self.mgraph with the fewest number of
        states.
        """
        d = self.to_DFA()
        classes = d.minimize_classes()
        # d.minimize()
        if self.debug:
            print 'classes', classes
        self.minimized_mgraph = condensation_multi(self.mgraph,classes)
        
        # state2node = lambda s: tuple(d.minimized_states[s])
        # # convert to a multigraph, since there could be multiple edges
        # # between any given pair of states now
        # self.minimized_mgraph = nx.MultiDiGraph()
        # for s in d.accepts:               # accepts are just indices
        #     self.minimized_mgraph.add_node( state2node(s) )
        # for s in d.accepts:                 # all accept state indices
        #     for a in self.symbols():
        #         t = d.transitions[s,a]    # t is a state index
        #         if t in d.accepts:        # only add valid transitions
        #             self.minimized_mgraph.add_edge(state2node(s),
        #                                            state2node(t),label=a)            

     
    def __repr__(self):
        if hasattr(self,'minimized_mgraph'):
            s = ("SoficProcessor on %i symbols, with %i states and %i transitions"
                 % (self.num_symbols(),
                    self.minimized_mgraph.number_of_nodes(),
                    self.minimized_mgraph.number_of_edges()) )
        else:
            s = ("SoficProcessor on %i symbols, with %i states and %i transitions"
                 % (self.num_symbols(),
                    self.mgraph.number_of_nodes(),
                    self.mgraph.number_of_edges()) )
        return s


    def index_map_to_latex_graph(self):
        """
        Returns a string (in TikZ code) representing the original
        labeled matrix as a graph with submatrices on the edges.
        """
        latex = ""
        for e in self.symbol_graph.edges_iter():
            s = str(self.edge_matrices[e])
            s = string.replace(s,"H","")
            s = string.replace(s,"[","{")
            s = string.replace(s,"]","}")
            if e[0]==e[1]:
                bend = "loop right"
            else:
                bend = "bend right"
            latex += ( "\path (%d) edge[%s] node[Y] {\matmat%s} (%d);"
                       % (e[0]+1,bend,s,e[1]+1) )
            latex += '\n'
        return latex

    def sofic_shift_to_latex_graph(self,layout=graphviz_layout,just_symbols=False):
        """
        Returns a string (in TikZ code) representing the vertex or
        edge presentation # of the sofic shift given by running the
        processor.
        """
        latex = ""
        if hasattr(self,'minimized_mgraph'):
            G = self.minimized_mgraph
            nodes = G.nodes()
            pos = layout(G)
            for i in range(len(nodes)):
                nodestr = "$\\bullet$"
                latex += ( "\\node (%d) at (%f,%f) {%s};\n"
                           % (i,pos[nodes[i]][0],pos[nodes[i]][1],nodestr) )

            for u,v,d in G.edges(data=True):
                latex += ( "\\path (%d) edge node[midway,fill=white,inner sep=1pt]{%d} (%d);\n"
                           % (nodes.index(u),d['label']+1,nodes.index(v)) )
        else:
            G = self.mgraph
            nodes = G.nodes()
            pos = layout(G)
            for i in range(len(nodes)):
                n = nodes[i]
                if just_symbols:
                    nodestr = str(n[0]+1) # Don't display the actual matrices
                else:
                    s = str(n[1])
                    s = string.replace(s,"H","")
                    s = string.replace(s,"[","{")
                    s = string.replace(s,"]","}")
                    nodestr = ( "$\\!\\left\\{%d, \\matmat%s\\right\\}\\!$"
                                % (n[0]+1,s) )
                latex += ( "\\node (%d-%d) at (%f,%f) {%s};\n"
                           % (n[0]+1,i,pos[n][0],pos[n][1],nodestr) )

            for e in G.edges_iter():
                latex += ( "\\path (%d-%d) edge (%d-%d);\n"
                           % (e[0][0]+1,nodes.index(e[0]),
                              e[1][0]+1,nodes.index(e[1])) )
        return latex

    
if __name__ == "__main__":

    import numpy
    
    generators = numpy.matrix( [[1, 0, 1],
                                [0,-1, 1],
                                [1,-1, 1]] )
    
    regions = { 0 : [0,1], 1 : [2] }

    adjmatrix = numpy.matrix( [[1,1],
                               [1,1]]
                              )


    sof = SoficProcessor(generators, regions, debug=True)
    sof.process()
    print sof
    print "Entropy of the even shift:", sof.entropy()
    print "Now minimizing..."
    sof.minimize()
    print sof
