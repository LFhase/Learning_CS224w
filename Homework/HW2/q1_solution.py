from __future__ import print_function
import numpy as np


def Q1_1():
    graph = []
    graph.append([])  # node 0
    graph.append([2, 3])  # node 1
    graph.append([1, 3, 4])
    graph.append([])
    graph.append([2, 7, 8])
    graph.append([])
    graph.append([2, 5, 9, 10])
    graph.append([4, 8])
    graph.append([])
    graph.append([5, 6, 8, 10])
    graph.append([])

    prob = [0.5] * 11
    prob[3] = 1.0
    prob[5] = 1.0
    prob[8] = 0.0
    prob[10] = 0.0

    it = 0
    change = True
    while change:
        print("========iteration %d========" % (it))
        for (i, prob_i) in enumerate(prob):
            print('(%d: %.2f)' % (i, prob[i]), end=',')

        change = False
        for (i, node) in enumerate(graph):
            prob_new = 0.0
            if len(node) == 0:
                continue
            for j in node:
                prob_new += prob[j] / len(node)
            if abs(prob_new - prob[i]) > 1e-6:
                prob[i] = prob_new
                change = True
        it += 1
        # if it == 3:
        #     break


Q1_1()