import snap
import numpy as np
import matplotlib.pyplot as plt

def load_graph(name):
    '''
    Helper function to load graphs.
    Use "epinions" for Epinions graph and "email" for Email graph.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "epinions":
        G = snap.LoadEdgeList(snap.PNGraph, "soc-Epinions1.txt", 0, 1)
    elif name == 'email':
        G = snap.LoadEdgeList(snap.PNGraph, "email-EuAll.txt", 0, 1)   
    else: 
        raise ValueError("Invalid graph: please use 'email' or 'epinions'.")
    return G

def q1_1():
    '''
    You will have to run the inward and outward BFS trees for the 
    respective nodes and reason about whether they are in SCC, IN or OUT.
    You may find the SNAP function GetBfsTree() to be useful here.
    '''
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 2018, compare sizes 
    #and comment on where node 2018 lies.
    G = load_graph("email")
    #Your code here:
    
    
    
    
    
    
    ##########################################################################
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 224, compare sizes 
    #and comment on where node 224 lies.
    G = load_graph("epinions")
    #Your code here:
    
    
    
    
    
    
    ##########################################################################

    print '2.1: Done!\n'


def q1_2():
    '''
    For each graph, get 100 random nodes and find the number of nodes in their
    inward and outward BFS trees starting from each node. Plot the cumulative
    number of nodes reached in the BFS runs, similar to the graph shown in 
    Broder et al. (see Figure in handout). You will need to have 4 figures,
    one each for the inward and outward BFS for each of email and epinions.
    
    Note: You may find the SNAP function GetRndNId() useful to get random
    node IDs (for initializing BFS).
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    
    
    
    
    
    
    








    
    
    
    
    
    
    
    
    ##########################################################################
    print '2.2: Done!\n'

def q1_3():
    '''
    For each graph, determine the size of the following regions:
        DISCONNECTED
        IN
        OUT
        SCC
        TENDRILS + TUBES
        
    You can use SNAP functions GetMxWcc() and GetMxScc() to get the sizes of 
    the largest WCC and SCC on each graph. 
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################
    print '2.3: Done!\n' 

def q1_4():
    '''
    For each graph, calculate the probability that a path exists between
    two nodes chosen uniformly from the overall graph.
    You can do this by choosing a large number of pairs of random nodes
    and calculating the fraction of these pairs which are connected.
    The following SNAP functions may be of help: GetRndNId(), GetShortPath()
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################
    print '2.4: Done!\n'
    
if __name__ == "__main__":
    q1_1()
    q1_2()
    q1_3()
    q1_4()
    print "Done with Question 2!\n"