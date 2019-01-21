from glob import *
import os
import re
import gzip

from data_structures import *

import gzip

def open_file(filename):
    if filename.endswith('gz'):
        return gzip.open(filename)
    else:
        return open(filename)

def read_tsp_file(file_name):
    city_coordinates = []
    with open_file(file_name) as f:
        edge_weight_matrix = []
        raw_matrix = []
        graph_type = ''
        edge_weight_format = ''
        dimension = 0
        begin_coordinates = False
        
        for line in f:
            # read n cities
            result = re.match('DIMENSION(\s)?: (\d+)', line)
            if not result is None:
                dimension = int(result.group(2))

            result = re.match('EDGE_WEIGHT_TYPE(\s)?:(\s)?(\w+)', line)
            # store graph type
            if not result is None:
                graph_type = result.group(3)

            result = re.match('EDGE_WEIGHT_FORMAT(\s)?:(\s)?(\w+)', line)
            # store graph type
            if not result is None:
                edge_weight_format = result.group(3)
                
            result = re.match('NODE_COORD_SECTION(\s)?|DISPLAY_DATA_SECTION(\s)?', line)
            # store graph type
            if not result is None:
                begin_coordinates = True
                
            if edge_weight_format == 'FULL_MATRIX':
                whiteSpaceRegex = "\\s"
                words = filter(None, re.split(whiteSpaceRegex, line))

                if len(words) >= dimension:
                    edge_weight_line = []
                    for weight in words:
                        edge_weight_line.append(int(weight))
                    edge_weight_matrix.append(edge_weight_line)
                    
            elif edge_weight_format == 'LOWER_DIAG_ROW':
                whiteSpaceRegex = "\\s"
                words = filter(None, re.split(whiteSpaceRegex, line))
                
                if len(words) > 0 and not (re.match('(\d)', words[0]) is None) and not begin_coordinates:
                    for i in words:
                        raw_matrix.append(int(i))
            whiteSpaceRegex = "\\s"
            words = filter(None, re.split(whiteSpaceRegex, line))
            if len(words) == 3 and begin_coordinates:
                try:
                    city_coordinates.append((float(words[1]), float(words[2])))
                except ValueError:
                    pass

        
        if graph_type == 'EUC_2D':
            assert len(city_coordinates) == dimension
            return EuclideanGraph(city_coordinates)
            
        if graph_type == 'ATT':
            assert len(city_coordinates) == dimension
            return PseudoEuclideanGraph(city_coordinates)
        
        if graph_type == 'GEO':
            assert len(city_coordinates) == dimension
            return GeographicalGraph(city_coordinates)
        
        if graph_type == 'EXPLICIT':
            if edge_weight_format == 'LOWER_DIAG_ROW':
                edge_weight_matrix = [[0 for i in range(dimension)] for j in range(dimension)]
                #####ERROR HERE!!!
                raw_matrix.reverse()
                for j in range(dimension):
                    for i in range(j + 1):
                        edge_weight_matrix[i][j] = raw_matrix.pop()
                        edge_weight_matrix[j][i] = edge_weight_matrix[i][j]
                
                
            #print len(city_coordinates)
            #print dimension
            #print city_coordinates
            "Verify file consistence"
            assert len(city_coordinates) == dimension
            assert len(edge_weight_matrix) == dimension
            assert len(edge_weight_matrix[0]) == dimension
            
            "Return a graph built with an edge weight matrix"
            return EdgeMatrixGraph(city_coordinates, edge_weight_matrix)

def read_tsp_files_from_directory(dir_name):
    graphs = []
    for root, directories, file_names in os.walk(dir_name):
        for file_name in file_names:
            if file_name.endswith('.tsp'):
                graphs.append(read_tsp_file(file_name))



