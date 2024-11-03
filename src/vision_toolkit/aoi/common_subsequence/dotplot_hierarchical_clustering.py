# -*- coding: utf-8 -*-


from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from treelib import Tree


class Dot:
    def __init__(self, i, j, w, item):
        self.i = i
        self.j = j
        self.w = w
        self.item = item


class DotPlot:
    def __init__(self, seqs, freq_thrs, distance_thrs, r2_thrs):
        # Keep sequences
        self.S = seqs

        # Parameters
        self.f_t = freq_thrs
        self.d_t = distance_thrs
        self.r2_t = r2_thrs

        # Compute item frequencies
        self.item_freq = self.comp_freq()

        # Generate DotPlot representation
        self.dot_repr = self.comp_repr()

        # Generate matching sequences for each pair of sequences
        self.matching_seqs = self.comp_lcs_matching_seq()

    def comp_freq(self):
        S_c = ""

        for i in range(len(self.S)):
            S_c += self.S[i]

        count = Counter(S_c)

        freq_d = dict()
        n = len(S_c)

        for key in count.keys():
            freq = count[key] / n
            freq_d.update({key: freq})

        return freq_d

    def comp_repr(self):
        n = len(self.S)
        f_t = self.f_t

        i_f = self.item_freq
        _repr = dict()

        def comp_dots(s1, s2):
            n_1 = len(s1)
            n_2 = len(s2)
            dots = []

            for i in range(n_1):
                for j in range(n_2):
                    if s1[i] == s2[j]:
                        w = i_f[s1[i]]
                        if w >= f_t:
                            dot = Dot(i, j, w, s1[i])
                            dots.append(dot)

            return dots

        for j in range(n):
            for i in range(j):
                key = "{key1},{key2}".format(key1=str(i), key2=str(j))
                dots = comp_dots(self.S[i], self.S[j])
                _repr.update({key: dots})

        return _repr

    def lcs(self, s1, s2):
        n_1 = len(s1)
        n_2 = len(s2)

        d_mat = np.zeros((n_1 + 1, n_2 + 1))

        for i in range(1, n_1 + 1):
            for j in range(1, n_2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    d_mat[i, j] = d_mat[i - 1, j - 1] + 1

                else:
                    m_ = max(d_mat[i, j - 1], d_mat[i - 1, j])
                    d_mat[i, j] = m_

        def backtrack(distance_mat, common_subseq, s1, s2, i, j):
            if i == 0 or j == 0:
                return common_subseq

            if s1[i - 1] == s2[j - 1]:
                common_subseq.insert(0, s1[i - 1])
                return backtrack(distance_mat, common_subseq, s1, s2, i - 1, j - 1)

            if distance_mat[i, j - 1] > distance_mat[i - 1, j]:
                return backtrack(distance_mat, common_subseq, s1, s2, i, j - 1)

            return backtrack(distance_mat, common_subseq, s1, s2, i - 1, j)

        c_s = backtrack(d_mat, list(), s1, s2, n_1, n_2)

        return c_s

    def comp_lcs_matching_seq(self):
        n = len(self.S)
        matching_seqs = np.zeros((n, n))

        for j in range(n):
            for i in range(j):
                c_s = self.lcs(self.S[i], self.S[j])

                matching_seqs[i, j] = len(c_s)

        matching_seqs = matching_seqs + matching_seqs.T

        return matching_seqs

    def comp_reg_matching_seq(self):
        n = len(self.S)
        matching_seqs = np.zeros((n, n))

        # Distance between point (x1,y1) and line given by the
        # equation y = ax + b
        def shortest_distance(x1, y1, a, b):
            d = abs((y1 - a * x1 - b)) / (np.sqrt(a**2 + 1))

            return d

        for j in range(n):
            for i in range(j):
                key = "{key1},{key2}".format(key1=str(i), key2=str(j))
                dots_st1 = self.dot_repr[key]

                # Compute first regression
                x_st1 = np.array([dot.j for dot in dots_st1])
                y_st1 = np.array([dot.i for dot in dots_st1])

                # r-squared computation
                cor = np.corrcoef(x_st1, y_st1)[0, 1]
                r2 = cor**2

                if r2 < self.r2_t:
                    m_s = list()
                    matching_seqs.update({key: m_s})

                else:
                    c_1 = np.polyfit(x_st1, y_st1, 1)
                    dists_st1 = np.array(
                        [
                            shortest_distance(x_st1[k], y_st1[k], c_1[0], c_1[1])
                            for k in range(len(x_st1))
                        ]
                    )

                    # Compûte distance threshold
                    d_mu = np.mean(dists_st1)
                    d_std = np.std(dists_st1)

                    thrs = d_mu + d_std

                    # Identify data within the threshold
                    dots_st2 = [
                        dots_st1[k] for k in range(len(dots_st1)) if dists_st1[k] < thrs
                    ]

                    # Compute second regression
                    x_st2 = np.array([dot.j for dot in dots_st2])
                    y_st2 = np.array([dot.i for dot in dots_st2])

                    plt.scatter(x_st2, y_st2)

                    c_2 = np.polyfit(x_st2, y_st2, 1)

                    dists_st2 = np.array(
                        [
                            shortest_distance(x_st2[k], y_st2[k], c_2[0], c_2[1])
                            for k in range(len(x_st2))
                        ]
                    )

                    dots_st3 = [
                        dots_st2[k]
                        for k in range(len(dots_st2))
                        if dists_st2[k] < self.d_t
                    ]

                    # Compute matching sequence
                    m_s = [dot.item for dot in dots_st3]

                    matching_seqs[i, j] = len(m_s)

        matching_seqs = matching_seqs + matching_seqs.T

        return matching_seqs


class Node:
    def __init__(self, sequence, children):
        self.seq = sequence
        self.children = children


class HierarchicalClustering:
    def __init__(self, seqs, freq_thrs, distance_thrs=1.0, r2_thrs=0.0001):
        dotplot = DotPlot(seqs, freq_thrs, distance_thrs, r2_thrs)

        self.S = seqs

        self.matching_seqs = dotplot.matching_seqs

    def process(self):
        S = self.S

        matching_seqs = self.matching_seqs
        t_matching_seqs = np.sum(matching_seqs, axis=1)

        available = {}

        for i in range(len(S)):
            available.update({i: Node(S[i], list())})

        while len(available.keys()) > 1:
            _max = 0
            o_pair = None

            for i in available.keys():
                for j in available.keys():
                    if matching_seqs[i, j] > _max:
                        _max = matching_seqs[i, j]
                        o_pair = [i, j]

            # print(o_pair)
            # print(t_matching_seqs)

            if t_matching_seqs[o_pair[0]] > t_matching_seqs[o_pair[1]]:
                parent = Node(
                    S[o_pair[0]], [available[o_pair[0]], available[o_pair[1]]]
                )

                del available[o_pair[0]]
                del available[o_pair[1]]

                available.update({o_pair[0]: parent})

            else:
                parent = Node(
                    S[o_pair[1]], [available[o_pair[0]], available[o_pair[1]]]
                )

                del available[o_pair[0]]
                del available[o_pair[1]]

                available.update({o_pair[1]: parent})

            # print("available " + str(available.keys()))

        root = available[list(available)[0]]
        tree = Tree()

        def make_tree(node, parent, tree):
            if parent == "":
                tree.create_node(node.seq, node.seq + parent)

            else:
                tree.create_node(node.seq, node.seq + parent, parent)

            if node.children != []:
                make_tree(node.children[0], node.seq + parent, tree)
                make_tree(node.children[1], node.seq + parent, tree)

        make_tree(root, "", tree)
        tree.show(line_type="ascii-em")


sequ = "XMJYXUZ"
seqv = "MZJXJYAWXU"
seqw = "MZJAJYYAWXU"
seqx = "AWXJYAW"
seqy = "MZJAWXU"
seqz = "XMJYAUZ"


# print(M)

S = [sequ, seqv, seqw, seqx, seqy, seqz]
freq_threshold = 0.01


dp = HierarchicalClustering(S, freq_threshold)
dp.process()
