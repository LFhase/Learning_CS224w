import snap
import numpy as np
import matplotlib.pyplot as plt


def load_graph_weights(path="hw1-q2.graph"):

    g = snap.TUNGraph.Load(snap.TFIn(path))

    return g


def extract_basic_feature(node, graph):

    v = [node.GetDeg()]
    tot_edges = 0
    nbrs = []
    for i in range(v[0]):
        nbrs.append(graph.GetNI(node.GetNbrNId(i)))
        tot_edges += nbrs[-1].GetDeg()
    inner_edges = 0
    for i in range(v[0]):
        for j in range(i):
            inner_edges += nbrs[i].IsInNId(nbrs[j].GetId())
    v.append(inner_edges)
    v.append(tot_edges - 2 * inner_edges)

    return np.array(v)


def cal_cos_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cal_initial_feature(graph):
    v_mat = []
    for node in graph.Nodes():
        v_mat.append(extract_basic_feature(node, graph))
    return np.array(v_mat)


def aggregrate(v, graph):
    v_ori = v.copy()
    n, f = v.shape
    v_mean = np.zeros(v.shape)
    v_sum = np.zeros(v.shape)
    for node in graph.Nodes():
        deg = node.GetDeg()
        for i in range(deg):
            v_sum[node.GetId()] += v_ori[node.GetNbrNId(i)]
        v_mean[node.GetId()] = v_sum[node.GetId()] / deg if deg != 0 else 0
    # n x 3*f
    return np.concatenate((v_ori, v_mean, v_sum), axis=1)


def q2_1():
    print("============")
    g = load_graph_weights()
    v_9 = extract_basic_feature(g.GetNI(9), g)
    v_mat = cal_initial_feature(g)

    cos_sim = [(i, cal_cos_sim(v_9, v)) for i, v in enumerate(v_mat)]
    cos_sim.sort(key=lambda y: y[1], reverse=True)
    print("Feature vector of node 9 is: ", v_9)
    print("Top 5 nodes at most similar to node 9 are: ")
    for node in cos_sim[1:6]:
        print(node, end=" ")
    print("\n============")


q2_1()


def q2_2():
    print("============")
    g = load_graph_weights()
    v_mat = cal_initial_feature(g)
    for k in range(2):
        v_mat = aggregrate(v_mat, g)
    v_9 = v_mat[9]

    cos_sim = [(i, cal_cos_sim(v_9, v)) for i, v in enumerate(v_mat)]
    cos_sim.sort(key=lambda y: y[1], reverse=True)
    print("Feature vector of node 9 is: ", v_9)
    print("Top 5 nodes at most similar to node 9 are: ")
    for node in cos_sim[1:6]:
        print(node, end=" ")
    print("\n============")


q2_2()


def get_2_nd_subgraph(node, graph):

    v = set()
    v.add(int(node))
    cur_v = set()
    cur_v.add(int(node))

    for k in range(2):
        nxt_nbrs = set()
        for u in cur_v:
            u = graph.GetNI(u)
            for i in range(u.GetDeg()):
                v.add(u.GetNbrNId(i))
                nxt_nbrs.add(u.GetNbrNId(i))
        cur_v = nxt_nbrs.copy()
    V = snap.TIntV()
    for node in v:
        V.Add(node)
    return V


def find_node(lower, upper, data):
    choice = -1
    while True:
        choice = np.random.randint(len(data), size=1)[0]
        if data[choice][1] >= lower and data[choice][1] <= upper:
            break
    return choice


def q2_3():
    print("============")
    g = load_graph_weights()
    v_mat = cal_initial_feature(g)
    for k in range(2):
        v_mat = aggregrate(v_mat, g)
    v_9 = v_mat[9]

    cos_sim = [(i, cal_cos_sim(v_9, v)) for i, v in enumerate(v_mat)]
    cos_sim.sort(key=lambda y: y[1], reverse=True)
    cos_sims = [u[1] for u in cos_sim]
    plt.hist(cos_sims, bins=20)
    plt.title("distribution of cosine similarity")
    plt.show()

    subg_1 = get_2_nd_subgraph(find_node(0, 0.05, cos_sim), g)
    subg_1_h = snap.TIntStrH()
    subg_1_h[subg_1[0]] = "blue"
    subg_1 = snap.ConvertSubGraph(snap.PUNGraph, g, subg_1)
    snap.DrawGViz(subg_1, snap.gvlNeato, "subgraph_1.png", "subgraph 1", True, subg_1_h)

    subg_2 = get_2_nd_subgraph(find_node(0.4, 0.45, cos_sim), g)
    subg_2_h = snap.TIntStrH()
    subg_2_h[subg_2[0]] = "blue"
    subg_2 = snap.ConvertSubGraph(snap.PUNGraph, g, subg_2)
    snap.DrawGViz(subg_2, snap.gvlNeato, "subgraph_2.png", "subgraph 2", True, subg_2_h)

    subg_3 = get_2_nd_subgraph(find_node(0.6, 0.65, cos_sim), g)
    subg_3_h = snap.TIntStrH()
    subg_3_h[subg_3[0]] = "blue"
    subg_3 = snap.ConvertSubGraph(snap.PUNGraph, g, subg_3)
    snap.DrawGViz(subg_3, snap.gvlNeato, "subgraph_3.png", "subgraph 3", True, subg_3_h)

    subg_4 = get_2_nd_subgraph(find_node(0.9, 0.95, cos_sim), g)
    subg_4_h = snap.TIntStrH()
    subg_4_h[subg_4[0]] = "blue"
    subg_4 = snap.ConvertSubGraph(snap.PUNGraph, g, subg_4)
    snap.DrawGViz(subg_4, snap.gvlNeato, "subgraph_4.png", "subgraph 4", True, subg_4_h)

    print("============")


q2_3()