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


def plot_reachablity(g, name=""):
    tot_nodes = g.GetNodes()
    sampled_nodes = np.random.randint(tot_nodes, size=100)
    nodes_in_cnt = []
    nodes_out_cnt = []

    for ni in sampled_nodes:
        bfs_in = snap.GetBfsTree(g, int(ni), False, True)
        nodes_in_cnt.append(bfs_in.GetNodes())
        bfs_out = snap.GetBfsTree(g, int(ni), True, False)
        nodes_out_cnt.append(bfs_out.GetNodes())

    plt.title("Reachability Using Links " + name)
    plt.xlabel("Frac. of Starting Nodes")
    plt.ylabel("Number of Nodes Reached")

    X = np.array(range(1, 101)) / 100
    Y = np.array(sorted(nodes_in_cnt))
    plt.plot(X, Y, label="in link")
    for (i, (x, y)) in enumerate(zip(X, Y)):
        if i + 1 < 100 and Y[i + 1] - y > 1000:
            plt.text(x + 0.03, y + 10, x)
    Y = np.array(sorted(nodes_out_cnt))
    plt.plot(X, Y, label="out link")
    for (i, (x, y)) in enumerate(zip(X, Y)):
        if i + 1 < 100 and Y[i + 1] - y > 1000:
            plt.text(x + 0.03, y + 10, x)
    plt.legend()
    plt.show()


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
    email_in = snap.GetBfsTree(G, 2018, False, True)
    email_out = snap.GetBfsTree(G, 2018, True, False)
    print("Total nodes of email: ", G.GetNodes())
    print("IN of 2018 in email: ", email_in.GetNodes())
    print("OUT of 2018 in email: ", email_out.GetNodes())
    ##########################################################################

    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 224, compare sizes
    #and comment on where node 224 lies.
    G = load_graph("epinions")
    #Your code here:
    ep_in = snap.GetBfsTree(G, 224, False, True)
    ep_out = snap.GetBfsTree(G, 224, True, False)
    print("Total nodes of epinions: ", G.GetNodes())
    print("IN of 224 in epinions: ", ep_in.GetNodes())
    print("OUT of 224 in epinions: ", ep_out.GetNodes())
    ##########################################################################

    print('1.1: Done!\n')


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
    G = load_graph("email")
    plot_reachablity(G, "email")
    G = load_graph("epinions")
    plot_reachablity(G, "epinions")
    ##########################################################################
    print('1.2: Done!\n')


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
    G = load_graph("email")
    print("Total nodes of email: ", G.GetNodes())
    wcc = snap.GetMxWcc(G)
    print("Size of MxWcc in email: ", wcc.GetNodes())
    scc = snap.GetMxScc(G)
    print("Size of MxScc in email: ", scc.GetNodes())
    scc_nodes = set()
    for ni in scc.Nodes():
        scc_nodes.add(ni.GetId())
    In = set()
    Out = set()
    for ni in scc.Nodes():
        ni_in = snap.GetBfsTree(G, ni.GetId(), False, True)
        for nni in ni_in.Nodes():
            if nni.GetId() not in scc_nodes:
                In.add(nni.GetId())
        ni_out = snap.GetBfsTree(G, ni.GetId(), True, False)
        for nni in ni_out.Nodes():
            if nni.GetId() not in scc_nodes:
                Out.add(nni.GetId())
        break
    print("Size of IN in email: ", len(In))
    print("Size of OUT in email: ", len(Out))
    print("Size of TT in email: ", wcc.GetNodes() - scc.GetNodes() - len(In) - len(Out))
    print("Size of DISCONNECT in email: ", G.GetNodes() - wcc.GetNodes())

    G = load_graph("epinions")
    print("Total nodes of epinions: ", G.GetNodes())
    wcc = snap.GetMxWcc(G)
    print("Size of MxWcc in epinions: ", wcc.GetNodes())
    scc = snap.GetMxScc(G)
    print("Size of MxScc in epinions: ", scc.GetNodes())
    scc_nodes = set()
    for ni in scc.Nodes():
        scc_nodes.add(ni.GetId())
    In = set()
    Out = set()
    for ni in scc.Nodes():
        ni_in = snap.GetBfsTree(G, ni.GetId(), False, True)
        for nni in ni_in.Nodes():
            if nni.GetId() not in scc_nodes:
                In.add(nni.GetId())
        ni_out = snap.GetBfsTree(G, ni.GetId(), True, False)
        for nni in ni_out.Nodes():
            if nni.GetId() not in scc_nodes:
                Out.add(nni.GetId())
        break
    print("Size of IN in epinions: ", len(In))
    print("Size of OUT in epinions: ", len(Out))
    print("Size of TT in epinions: ", wcc.GetNodes() - scc.GetNodes() - len(In) - len(Out))
    print("Size of DISCONNECT in epinions: ", G.GetNodes() - wcc.GetNodes())
    ##########################################################################
    print('1.3: Done!\n')


def cal_con_prob(g, scc_set, In, Out):
    sample_points = []
    probs = []
    samples = np.random.randint(g.GetNodes(), size=g.GetNodes())
    connected_cnt = 0
    for i in range(1, int(g.GetNodes() * 0.03)):
        sample_points.append(i / g.GetNodes())
        probs.append(connected_cnt / i)

        # if len(probs) > 100 and abs(probs[-1] - probs[i - 100]) < 1e-6:
        #     break

        u = samples[i << 1]
        v = samples[i << 1 | 1]
        if (u in scc_set or u in In) and (v in scc_set or v in Out):
            connected_cnt += 1
            continue
        if (u in Out and (v in In or v in scc_set)) or (v in In and (u in Out or u in scc_set)):
            continue
        try:
            g_out = snap.GetBfsTree(g, int(u), True, False)
        except Exception as e:
            print(e)
            i -= 1
            continue
        for nii in g_out.Nodes():
            if nii.GetId() == v:
                connected_cnt += 1
                break

    return sample_points, probs


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

    G = load_graph("email")
    scc = snap.GetMxScc(G)
    scc_set = set()
    for ni in scc.Nodes():
        scc_set.add(ni.GetId())
    In = set()
    Out = set()
    for ni in scc.Nodes():
        ni_in = snap.GetBfsTree(G, ni.GetId(), False, True)
        for nni in ni_in.Nodes():
            if nni.GetId() not in scc_set:
                In.add(nni.GetId())
        ni_out = snap.GetBfsTree(G, ni.GetId(), True, False)
        for nni in ni_out.Nodes():
            if nni.GetId() not in scc_set:
                Out.add(nni.GetId())
        break

    sample_points, probs_email = cal_con_prob(G, scc_set, In, Out)
    plt.plot(sample_points, probs_email, label="email")
    plt.text(sample_points[-1] + 0.01, probs_email[-1] + 0.01, probs_email[-1])

    G = load_graph("epinions")
    scc = snap.GetMxScc(G)
    scc_set = set()
    for ni in scc.Nodes():
        scc_set.add(ni.GetId())
    In = set()
    Out = set()
    for ni in scc.Nodes():
        ni_in = snap.GetBfsTree(G, ni.GetId(), False, True)
        for nni in ni_in.Nodes():
            if nni.GetId() not in scc_set:
                In.add(nni.GetId())
        ni_out = snap.GetBfsTree(G, ni.GetId(), True, False)
        for nni in ni_out.Nodes():
            if nni.GetId() not in scc_set:
                Out.add(nni.GetId())
        break

    sample_points, probs_ep = cal_con_prob(G, scc_set, In, Out)
    plt.plot(sample_points, probs_ep, label="epinions")
    plt.text(sample_points[-1] + 0.01, probs_ep[-1] + 0.01, probs_ep[-1])

    plt.title("Connectivity Probability")
    plt.xlabel("Frac. of Sampled Pairs over Graph Size")
    plt.ylabel("Connectivity Probability")
    plt.legend()
    plt.show()
    ##########################################################################
    print('1.4: Done!\n')


if __name__ == "__main__":
    #q1_1()
    #q1_2()
    #q1_3()
    q1_4()
    print("Done with Question 1!\n")