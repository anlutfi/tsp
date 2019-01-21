from data_structures import *
import random
import gnuplot
from solvers import *
import sys

# Some simple functions for testing if the TSPs are doing the right thing.

TEST_GRAPH = None

def test_graph():
    return TEST_GRAPH
        
def benchmark_tsp(to_test, graph = None, n = None, GraphType = EuclideanGraph, log_to_file = False, always_plot = False):
    
    global TEST_GRAPH
    
    if graph is not None:
        TEST_GRAPH = graph
    elif n is not None:
        TEST_GRAPH = GraphType.random_graph(n)
    else:
        print "Either pass a graph or a graph dimension."
        return
    
    results = [tsp.run(TEST_GRAPH) for tsp in to_test]
    
    max_name_len = 0
    max_iter_len = 0
    max_lb_len = 0
    wrong_result = False
    for (i,result) in enumerate(results):
        max_name_len = max(max_name_len, len(result.alg))
        max_iter_len = max(max_iter_len, len(str(result.iter_count)))
        max_lb_len = max(max_lb_len, len(str(result.lower_bound)))
        if i > 0 and results[i].cost() != results[i-1].cost():
            wrong_result = True
    
    def print_result(log, result):
        log.write(
                            ("%" + str(max_name_len) + "s")%result.alg + 
            ("--- iter count: %" + str(max_iter_len) + "d")%result.iter_count +
            ("--- lower bound: %" + str(max_lb_len) + "d")%result.lower_bound +
            ("--- tour cost:  %f")%result.cost() + "\n"
        )
    
    if wrong_result and log_to_file:
        with open("log_error.txt", "a") as log:
            log.write("Vertices: " + str(TEST_GRAPH.city_coordinates))
            if GraphType == EdgeMatrixGraph:
                log.write("\n\nEdges: " + str(TEST_GRAPH.edge_matrix))
            log.write("\n\n")
            for result in results:
                print_result(log, result)
            log.write("\n================================================================\n\n")
    else:
        for result in results:
            if wrong_result or always_plot:
                gnuplot.print_tour(result.get_tour())
            print_result(sys.stdout, result)
        print ""
              
    return result
    
def infinite_test(*args, **kwargs):
    while (True):
        benchmark_tsp(*args, **kwargs)
