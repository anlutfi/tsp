#! /usr/bin/python

# Tests

from solvers import *
from read_instances import *
from gnuplot import *
import signal

import sys

def usage():
   print "usage: ./test.py tsp_instance_file solvertype args" 

def main():

    if len(sys.argv) <= 2:
        return usage()
    
    (_, filename, solvertype) = sys.argv[:3]
    args = [float(x) for x in sys.argv[3:]]
    
    solver_classes = {
        'stupid': StupidBackTracker,
        'dynamic': DynamicProgrammingTSP,
        'simple': ShortestPathBackTracker,
        'qroute': QBackTracker,
        'mst': MSTBackTracker,
        'heldkarp': HeldKarpBackTracker,
    }
    
    solver_cls = solver_classes.get( solvertype.lower() )
    
    if solver_cls is None:
        print "unkown solver type"
        print "known solvers:"
        for n in solver_classes.iterkeys():
            print "  ", n
        return
    
    solver = solver_cls(*args)
    
    graph = read_tsp_file('./instancias/'+filename)
    
    
    def alarm_handler(signum, frame):
        raise Timeout
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(60 * 30)
    
    result = solver.run(graph)
    
    with open("./results/"+filename+'.txt', "a") as out:
        result.print_result(out)
    

if __name__ == '__main__':
    main()
