"""
utils.py

Author: Jesse Berwald

Opened: Feb. 15, 2012

Tools for importing from matlab and other formats.
"""
import networkx as nx
import numpy as np
import cPickle as pkl
import scipy.io as spio
from scipy import sparse

def load_numpy( fname ):
    """
    Returns numpy array stored as .npy file
    """
    return np.matrix( np.load( fname ) )

def loadtxt( fname, dtype=np.int ):
    """
    Returns numpy array stored as .npy file
    """
    return np.matrix( np.loadtxt( fname, dtype=dtype ) )

def load_matlab_matrix( matfile, matname=None ):
    """
    Wraps scipy.io.loadmat.

    If matname provided, returns np.ndarray representing the index
    map. Otherwise, the full dict provided by loadmat is returns.
    """
    if not matname:
        out = spio.loadmat( matfile )
        mat = _extract_mat( out )
        # if mat is a sparse matrix, convert it to numpy matrix
        try:
            mat = np.matrix( mat.toarray() )
        except AttributeError:
            mat = np.matrix( mat )
        return mat
    else:
        matdict = spio.loadmat( matfile )
        mat = matdict[ matname ]
        # if mat is a sparse matrix, convert it to numpy matrix
        try:
            mat = np.matrix( mat.toarray() )
        except AttributeError:
            mat = np.matrix( mat )
        return mat #np.matrix( mat[ matname ] )

def cell2dict( ca ):
    """
    Parameters:
    -----------

    ca : cell array from Matlab, loaded using scipy.io.loadmat(). See
    convert_matlab_gens().

    ca is a np.ndarray of type object. Every entry is an array. Eg., if

        A = array([1, [2 3],[],...]), then
        
        A[0] = array(1,dtype=int8) <-- dimensionless array,
        A[1] = array([2,3],dtype=int8),
        etc.

    Empty lists (see above example) are quietly ignored.

    Returns a Python dictionary
    """

    # gens is a list of arrays, of type uint8, of shape (1,n)
    # region (r) |--> gen map
    keymap = {}
    ca = ca.flatten().tolist()
    
    # Remember to shift all region labels and generator labels by (-1)
    # to align with Python 0-based indexing.
    for k,v in enumerate( ca ):
        try:
            keymap[ k ] = map(lambda x: x-1, v.flatten().tolist())
        except AttributeError:
            keymap[ k ] = map(lambda x: x-1, v[0].flatten().tolist())
    return keymap

