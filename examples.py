"""
sofic_processor.py

Opened: May 2018

Author: Rafael Frongillo

Some examples showing how to use the Sofic Processor
"""

from sofic_processor_standalone import SoficProcessor
import pickle
import numpy


print "EXAMPLE 1: the even shift"

matrix = numpy.matrix( [[1, 0, 1],
                        [0,-1, 1],
                        [1,-1, 1]] )
    
labels = { 0 : [0,1], 1 : [2] }

sof = SoficProcessor(matrix, labels, debug=True)
sof.process()
print sof
print "Entropy of the even shift:", sof.entropy()
print "Now minimizing..."
sof.minimize()
print sof




print "EXAMPLE 2: Henon map"
matrix = pickle.load(open('henon-4555-M.pkl','rb'))
labels = pickle.load(open('henon-4555-G.pkl','rb'))

sof = SoficProcessor(matrix, labels, debug=False)

sof.process()
print 'finished processing:', sof
#print 'entropy:', sof.entropy()

sof.take_periodic_closure()
print 'periodic closure:', sof

sof.minimize()
print "minimized:", sof
