import math

#
# Graphs
#

class _Graph(object):

    #Public properties:
    
    # n_vertex
    # city_coordinates

    #Common methods
        
    def vs(self):
        "Vertex iterator from 0 to n-1"
        return xrange(self.n_vertex)
        
    def has_vertex(self, v):
        return 0 <= v < self.n_vertex
    
    #Implementation dependent methods
    
    def e(self, v, w):
        '''The edge weight for edge (v,w)'''
        raise NotImplementedError
        
    def min_path_cost(self, v, w):
        '''The minimum path between two given vertices.'''
        #This is needed to compute tour estimates when the triangular inequality does not hold.
        raise NotImplementedError
        
    @classmethod
    def random_graph(Cls, n):
        '''Generate a random instance with n vertices'''
        raise NotImplementedError

class _ImplicitGraph(_Graph):
    '''Class for graphs with edges specified by distance functions.'''
    def __init__(self, city_coordinates):
        self.n_vertex = len(city_coordinates)
        self.city_coordinates = city_coordinates
        
        assert self.n_vertex > 0 
        
        if self.n_vertex > 300:
            self.e = self._e_dist
        
        else:
            self.e = self._e_matrix
        
            self.edge_matrix = [[0 for i in xrange(self.n_vertex)] for j in xrange(self.n_vertex)]
            for (i, (x1, y1)) in enumerate(self.city_coordinates):
                for (j, (x2, y2)) in enumerate(self.city_coordinates):
                    if i != j:
                        self.edge_matrix[i][j] = self._dist(x1, x2, y1, y2)
                        self.edge_matrix[j][i] = self.edge_matrix[i][j]
            
    def _e_matrix(self, v, w):
        assert self.has_vertex(v)
        assert self.has_vertex(w)
        return self.edge_matrix[v][w]
    
    def _e_dist(self, v, w):
        assert self.has_vertex(v)
        assert self.has_vertex(w)
        (x1, y1) = self.city_coordinates[v]
        (x2, y2) = self.city_coordinates[w]
        return self._dist(x1,x2,y1,y2)

    def min_path_cost(self, v, w):
        #Euclidean distances satisfy the triangle inequality.
        return self.e(v, w)

    @classmethod
    def random_graph(Cls, n):
        return Cls( _random_coordinates(n) )


class EuclideanGraph(_ImplicitGraph):
    '''Distances are the Euclidean distance between the coordinates (rounded to the closest integer)'''
    def _dist(self, x1, x2, y1, y2):
        return int(round(math.sqrt((x1-x2)**2 + (y1-y2)**2)))

class PseudoEuclideanGraph(_ImplicitGraph):
    '''Distances are the Euclidean distance between the coordinates (rounded up to the closest integer)'''
    def _dist(self, x1, x2, y1, y2):
        return int(math.ceil( math.sqrt(((x1-x2)**2 + (y1-y2)**2)/10.0) ))
        
PI = 3.141592
RRR = 6378.388 #Earth's radius in Km

class GeographicalGraph(_ImplicitGraph):
    '''Distances that take earth's curvature into consideration'''
    
    def _to_latlong(self, x):
        degrees   = int(x)
        minutes   = x - degrees
        return (PI * (degrees + (5.0 * minutes) / 3.0)) / 180.0
    
    def _dist(self, x1, x2, y1, y2):
        lat1 = self._to_latlong(x1)
        lat2 = self._to_latlong(x2)
        lng1 = self._to_latlong(y1)
        lng2 = self._to_latlong(y2)
        
        q1 = math.cos(lng1 - lng2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        
        return int(RRR * math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0 )

class EdgeMatrixGraph(_Graph):

    def __init__(self, city_coordinates, edge_matrix):
    
        self.n_vertex = len(city_coordinates)
        self.city_coordinates = city_coordinates
        self.edge_matrix = edge_matrix
        
        #Is the matrix square and symmetric?
        #Are all the edge weights non-negative?
        assert self.n_vertex > 0
        for i in self.vs():
            assert len(self.edge_matrix[i]) == self.n_vertex
            for j in self.vs():
                assert (self.edge_matrix[i][j] == self.edge_matrix[j][i]) and self.edge_matrix[i][j] >= 0
        
        #Find shortest paths using Floyd-Warshall
        self.pair_dist = [[self.e(i, j) for j in self.vs()] for i in self.vs()]
        for k in self.vs():
            for i in self.vs():
                for j in self.vs():
                    self.pair_dist[i][j] = min(
                        self.pair_dist[i][j],
                        self.pair_dist[i][k] + self.pair_dist[k][j]
                    )
    
    def e(self, v, w):
        assert self.has_vertex(v)
        assert self.has_vertex(w)
        return self.edge_matrix[v][w]
        
    def min_path_cost(self, v, w):
        assert self.has_vertex(v)
        assert self.has_vertex(w)
        return self.pair_dist[v][w]
    
    @classmethod
    def random_graph(Cls, n):
        return Cls( _random_coordinates(n), _random_symmetric_matrix(n) )

#
# Paths and Tours
#

class _Path(object):
    '''Immutable paths'''
    #This way we don't need to do pop() operations when backtracking
    #And can branch the same path in multiple directions.

    def v_list(self):
        vs = []
        p = self
        while p:
            vs.append(p.vertex)
            p = p.prev
        vs.reverse()
        return vs
    
    def contains(self, v):
        return self.is_used[v]
    
    def is_hamiltonian(self):
        return self.length == self.graph.n_vertex

    def path_cost(self):
        return self.cost

    def tour_cost(self):
        assert self.is_hamiltonian()
        return self.cost + self.graph.e(self.vertex, self.v0)

class InitialPath(_Path):
    '''An empty path with a single vertex'''
    def __init__(self, graph, v):
        self.graph = graph
        self.prev = None
        self.vertex = v
        
        assert self.graph.has_vertex(v)
        
        self.length = 1
        self.cost = 0
        self.v0 = v
        self.unused_vertices = tuple(v for v in self.graph.vs() if v != self.vertex)



class PathCons(_Path):
    '''The '''
    def __init__(self, prev, v):
        self.graph = prev.graph
        self.prev = prev
        self.vertex = v
    
        assert self.graph.has_vertex(v)
        assert v in self.prev.unused_vertices
        
        self.length = prev.length + 1
        self.cost = self.prev.cost + self.graph.e(v, self.prev.vertex)
        self.v0 = self.prev.v0
        self.unused_vertices = tuple(v for v in prev.unused_vertices if v != self.vertex)

####

import random

def _r(n):
    return 10 * random.randrange(0, 30)

def _random_coordinates(n):
    return [(_r(n), _r(n)) for x in range(n)]

def _random_symmetric_matrix(n):
    m = [[0 for i in xrange(n)] for j in xrange(n)]
    for i in xrange(n):
        for j in xrange(n):
            if i != j:
                m[i][j] = _r(n)
                m[j][i] = m[i][j]
    return m
