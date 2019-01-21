from data_structures import *
import time

class Timeout (Exception):
    pass

class TSP_Result(object):
    def __init__(self, solver, time_in_seconds):
        self.tour = solver.min_tour
        self.iter_count = solver.iter_count
        self.alg = solver.class_name()
        self.lower_bound = max(solver.first_lower_bound, 0)
        self.upper_bound = solver.min_tour_cost
        self.time_in_seconds = time_in_seconds
        
        assert solver is not None
        
        self.tour = solver.min_tour
        self.tour_cost = solver.min_tour_cost
        self.tour_it = solver.min_tour_it
        
        self.iter_count = solver.iter_count
        self.node_count = self.tour.length if self.tour else None
        
        
    def print_graph(self):
        return str(self.tour.graph.city_coordinates)
              
    def cost(self):
        return self.tour_cost 
        
    def get_tour(self):
        return self.tour
    
    def print_result(self, out):
        out.write( '\t'.join([
            str(self.alg),
            str(self.time_in_seconds),
            str(self.iter_count),
            str(self.tour_it),
            str(self.lower_bound),
            str(self.tour_cost),
        ]) + '\n')

class _TSPSolver(object):

    def class_name(self):
        raise NotImplementedError
    
    def run(self, graph):
        self.min_tour_cost = None
        self.min_tour = None
        self.min_tour_it = None
        self.first_lower_bound = None
        self.iter_count = 0 

        print "running..."
        start_time = time.time()

        try:
            self.solve(graph)
        except (Timeout, KeyboardInterrupt):
            pass
        
        end_time = time.time()
        
        result = TSP_Result(self, end_time - start_time)
        
        self.min_tour_cost = None
        self.min_tour = None
        self.min_tour_it = None
        self.last_lower_bound = None
        self.iter_count = None
        
        return result
    
    def is_smaller_than_upper_bound(self, cost):
        return (self.min_tour is None) or (cost < self.min_tour_cost)
        
    def present_solution(self, path):
        cost = path.tour_cost()
        if self.is_smaller_than_upper_bound(cost):
            self.min_tour      = path
            self.min_tour_cost = cost
            self.min_tour_it   = self.iter_count
      
#
# Dynamic Programming
#

class DynamicProgrammingTSP(_TSPSolver):
    
    def class_name(self):
        return "DynamicProgrammingTSP"
    
    def solve(self, graph):
        
        def relax(layer, p):
            #Relaxes states of a given layer if possible
            hp = (p.unused_vertices, p.vertex)
            pp = layer.get(hp)
            if pp is None or p.cost < pp.cost:
                layer[hp] = p
        
        # curr_layer[(S, v)] is the length of the shortest path of length `nv`
        # starting at `v0`, passing though the vertices not in `S` and stopping at vertex `v`
        
        curr_layer = {}
        self.iter_count += 1
        relax(curr_layer, InitialPath(graph, 0))
                
        for nv in xrange(1, graph.n_vertex):
            self.iter_count += 1
            next_layer = {}
            for path in curr_layer.itervalues():
                for v in path.unused_vertices:
                    relax(next_layer, PathCons(path, v))
            curr_layer = next_layer
        
        for path in curr_layer.itervalues():
            self.present_solution(path)

#
# Branch and bound solvers
#

class BackTracker(_TSPSolver):
    "Backtracking TSP search with prunning"
    
    def __init__(self):
        pass
    
    def solve(self, graph):
        
        p0 = InitialPath(graph, 0)
        p0.pargs = None
        
        #note: pargs is a hacky way to pass around information from previous runs that can be used to 
        #      make lower bound computations cheaper.
        
        self.stack = [ iter([(p0, None, None)]) ]
        
        def no_lb_iterator(path):
            return ( (PathCons(path,v), None, None) for v in path.unused_vertices )
        
        def with_lb_iterator(search_nodes):
            return ( (p,lb,pot) for (p,lb,pot) in search_nodes if self.is_smaller_than_upper_bound(lb) )
        
        while self.stack:
            it = self.stack[-1]
            
            try:
                (path, lb, pargs) = it.next()
            except StopIteration:
                self.stack.pop()
                continue
            
            if lb is None:
                self.iter_count += 1

            if path.is_hamiltonian():
                self.present_solution(path)
            
            else:
                
                if not self.should_prune() or path.length == path.graph.n_vertex - 1:
                    #Simple backtracking without when we get near the leaves
                    #Also, we must not run lower bounds on the leaves anyway (they make no sense)
                    self.stack.append( no_lb_iterator(path) )
                
                else:
                    
                    if self.first_lower_bound is None:
                        self.first_lower_bound = lb if lb is not None else self.lower_bound(path)[0]
                        
                    #Vertex choice heiristic:
                    #Backtrack to the children with the highest lowers bounds first
                    next_search_nodes = []
                    for v in path.unused_vertices:
                        self.iter_count += 1 #The number of lower-bound computations
                        
                        p = PathCons(path, v)
                        p.pargs = pargs
                        
                        next_lb, next_pargs = self.lower_bound(p)
                        next_search_nodes.append( (p, next_lb, next_pargs) )
                    
                    next_search_nodes.sort(key = lambda (p,lb,pa): lb)
                    self.stack.append( with_lb_iterator(next_search_nodes) )
                

    #Things to overwrite:


    def should_prune(self):
        return True
    
    def lower_bound(self):
        raise NotImplementedError    
                    
class StupidBackTracker(BackTracker):
    def class_name(self):
        return "StupidBackTracker"

    def should_prune(self):
        return False

class ShortestPathBackTracker(BackTracker):
    def class_name(self):
        return "ShortestPathBackTracker"
    
    def lower_bound(self, path):
        return (path.path_cost() + path.graph.min_path_cost(path.vertex, path.v0), None)


class QBackTracker(BackTracker):
    def class_name(self):
        return "QBackTracker"
        
    def lower_bound(self, path):
        #Relax the need for the path back to the start to use each vertex once,
        # but maintain the restriction of having to use (N-k) edges.
        
        graph = path.graph
        vs = path.vertex
        v0 = path.v0
        
        #A simple dynamic programming solution:
        
        #  curr[i] is the length of the shortest (non-simple) path
        #  from the current vertex to the i-th non-visited vertex,
        #  using `nv` inner vertices.
        
        #  next is the curr for the i+1 iteration
        
        curr = [path.graph.e(vs, v) for v in path.unused_vertices]
        for nv in range(1, path.graph.n_vertex - path.length):
            next = [None for w in path.unused_vertices]
            for (i, v) in enumerate(path.unused_vertices):
                if curr[i] is not None:
                    for (j, w) in enumerate(path.unused_vertices):
                        if v != w:
                            cost = curr[i] + path.graph.e(v, w)
                            if next[j] is None or cost < next[j]:
                                next[j] = cost
            curr = next

        cost = (
            path.path_cost() +
            min(curr[i] + path.graph.e(v, v0) for (i,v) in enumerate(path.unused_vertices))
        )
        
        return (cost, None)

def minimum_one_tree(path, edge_cost):

    #Returns the edges for the minimum spanning tree
    # induced by `allowed_vertices` together with the
    # shortest (distinct) edges connecting the end of `path` to the mst 
    # and the start of `path` to the MST
    
    #Note that this 1-tree is a graph connecting the current vertex to the starting vertex
    # using exactly (N-k) edges and reaching every non-visited vertex.
    #This relaxation reflects all the desired properties of actual tours back home, except
    # for the requirement of the graph being a path (every vertex having degree 2)

    graph = path.graph
    vs = path.vertex
    v0 = path.v0
    
    allowed_vertices = path.unused_vertices

    #Prim's algorithm

    #dist[v] is the cheapest edge cost from the v-th vertex to the mst
    # (only applicable to vertices outside the mst)
    
    #parent[i] describes the minimum spanning tree as a branching
    
    dist      = [None  for v in graph.vs()]
    parent    = [None  for v in graph.vs()]
    in_tree   = [False for v in graph.vs()]
    
    dist[allowed_vertices[0]] = 0
    
    for nv in xrange(len(allowed_vertices)):
        
        #Find closest reacheable vertex that doesn't belong to the MST
        v = None
        for w in allowed_vertices:
            if not in_tree[w] and dist[w] is not None:
                if v is None or dist[w] < dist[v]:
                    v = w

        #Add it to the tree
        in_tree[v] = True
    
        #Relax its neighbouring vertices
        for w in allowed_vertices:
            if not in_tree[w]:
                c = edge_cost(v, w)
                if dist[w] is None or c < dist[w]:
                    dist[w] = c
                    parent[w] = v
    
    enter_v = min( allowed_vertices,
                  key=(lambda v: edge_cost(vs, v)) )
    
    #Make sure we don't use the same edge to go to and back from the tree (gives better lower bounds)
    leave_v = min( (v for v in allowed_vertices if (vs != v0) or v != enter_v),
                  key=(lambda v: edge_cost(v, v0)) )

    tree_edges = [(v, parent[v]) for v in graph.vs() if parent[v] is not None]
    tree_edges.append( (vs, enter_v) )
    tree_edges.append( (leave_v, v0) )
    
    return tree_edges

class MSTBackTracker(BackTracker):
    def class_name(self):
        return "MSTBackTracker"
    
    def lower_bound(self, path):
        tree_edges = minimum_one_tree(path, path.graph.e)
        cost = path.path_cost() + sum(path.graph.e(v,w) for (v,w) in tree_edges)
        return (cost, None)

class HeldKarpBackTracker(BackTracker):
    def class_name(self):
        return "HeldKarpBackTracker"
    
    def __init__(self, MAX_ITER_WITHOUT_IMPROVEMENT=17, STEP_SIZE=20):
        self.MAX_ITER = int(MAX_ITER_WITHOUT_IMPROVEMENT)
        self.STEP_SIZE = int(STEP_SIZE)
    
    def lower_bound(self, path):
        #In this case, we try to make the MST returned by the last
        #solution to have as many vertices with degree 2 as possible.
        #(A tree with all degrees equal to 2 is a path)
        
        #To do so, we introduce a potential vector p_v for the non-visited vertices,
        #and change the edge costs according to 
        #  e'(v,w) = e(v,w) + p(v) + p(w)
        #This way, paths will be "unnafected", since they use each potential exacly once,
        #but trees can be punished if we give a large potential for the vertices
        #with degree >= 3.
        
        #What we therefore do, is try to find a potential vector that maximizes
        #  min(cost_mst) - 2*sum(potentials)

        graph = path.graph
        
        best_lower_bound = None

        potentials = [0 for v in graph.vs()]
        if path.pargs:
            for v in path.unused_vertices:
                potentials[v] = path.pargs[v]

        
        step = self.STEP_SIZE
        
        def edge_cost(v,w):
            return graph.e(v,w) + potentials[v] + potentials[w]
        
        last_increase = 0
        
        it = 1
        while (it - last_increase) < self.MAX_ITER:
        
            tree_edges = minimum_one_tree(path, edge_cost)
            
            degree = [0 for v in graph.vs()]
            for (v,w) in tree_edges:
                degree[v] += 1
                degree[w] += 1
            
            degree_excess = [0 for v in graph.vs()]
            for v in path.unused_vertices: degree_excess[v] = degree[v] - 2
        
            tree_cost = (
                path.path_cost() + 
                sum( graph.e(v,w) for (v,w) in tree_edges ) +
                sum( potentials[v]*degree_excess[v] for v in path.unused_vertices )
            )                
            
            #update lower bound
            if best_lower_bound is None or tree_cost > best_lower_bound:
                best_lower_bound = tree_cost
                last_increase = it                
                if not self.is_smaller_than_upper_bound(best_lower_bound):
                    break #will lead to pruning

            if all(e == 0 for e in degree_excess):
                break

            #update potentials
            for v in path.unused_vertices:
                potentials[v] += step*degree_excess[v]
            
            #Do not update step size, according to the original Held Karp paper.
            #This also allows us to use integer arythmetic.
                        
            it += 1

        
        return (best_lower_bound, potentials)
