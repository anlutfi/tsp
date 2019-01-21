import subprocess

def print_tour(tour):
    _print_points(True, tour)
    
def print_path(path):
    _print_points(False, path)
    
def _print_points(is_tour, points):
    vertex_coords = points.graph.city_coordinates
    line_coords = [vertex_coords[v] for v in points.v_list()]
    
    min_x = min(x for (x, y) in vertex_coords)
    max_x = max(x for (x, y) in vertex_coords)
    min_y = min(y for (x, y) in vertex_coords)
    max_y = max(y for (x, y) in vertex_coords)
    
    d = max(max_x - min_x, max_y - min_y)
    
    # IMPORTANT: Make sure that the gnuplot executable can be found in your $PATH.
    
    #On cshell (LABPOS), add the following to your .cshrc file:
    #    setenv PATH ${PATH}:${HOME}/gnuplot/bin
    
    #On bash, add the following to .bashrc:
    #    export PATH=$PATH:$HOME/gnuplot/bin
    
    #Note that environment variable updates only work for subprocesses
    # of the shell where you set the values in. Python will use the correct
    # path if it was set in the shell you lanch Python from, before you run Python.
    # (This is why you should add the changes to your rc file)
    
    gnuplot = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE )

    gnuplot.stdin.write("set size square\n")
    
    gnuplot.stdin.write("plot [%d:%d] [%d:%d] '-' with labels, '-' with lines\n"
                        %(min_x - 10, min_x + d + 10,
                          min_y - 10, min_y+ d + 10))

    for (i, (x, y)) in enumerate(vertex_coords):
        gnuplot.stdin.write("%d %d %d\n" % (x, y, i))
    gnuplot.stdin.write("e\n")

    for (x, y) in line_coords:
        gnuplot.stdin.write("%d %d\n" % (x, y))
    if is_tour:
        (x, y) = vertex_coords[points.v0]
        gnuplot.stdin.write("%d %d\n"% (x, y))
    gnuplot.stdin.write("e\n")

    return
