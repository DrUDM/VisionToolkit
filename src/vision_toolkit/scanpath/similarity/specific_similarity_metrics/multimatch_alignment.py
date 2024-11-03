# -*- coding: utf-8 -*-

import copy
from itertools import groupby
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.utils.velocity_distance_factory import absolute_angular_distance as aad
from vision_toolkit.visualization.scanpath.similarity.multimatch import (
    plot_multi_match, plot_simplification)


class MultiMatch:
    def __init__(self, input, config, id_1, id_2):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.s_1 = input[0].values
        self.s_2 = input[1].values

        self.vf_diag = config["screen_diagonal"]

        self.m_i = config["multimatch_simplification_iterations"]

        self.amp_thrs = config["multimatch_simplification_amplitude_threshold"]
        self.dur_thrs = config["multimatch_simplification_duration_threshold"]
        self.ang_thrs = config["multimatch_simplification_angular_threshold"]

        ## Compute simplified scanpaths, using the convention that the fixation
        ## is at the beginning of the saccade vector: when merging saccade vectors,
        ## the durations are merged at the beginning of the saccade vector
        self.s_1_s = self.simplification(self.s_1)
        self.s_2_s = self.simplification(self.s_2)

        ## Compute simplified saccade vectors from simplified scanpaths
        self.v_s_1 = self.s_1_s[0:2, 1:] - self.s_1_s[0:2, :-1]
        self.v_s_2 = self.s_2_s[0:2, 1:] - self.s_2_s[0:2, :-1]

        ## Compute distance matrix between simplified saccade vectors
        c_m = cdist(self.v_s_1.T, self.v_s_2.T, metric="euclidean")

        ## Compute alignments from Dijkstra optimal path
        self.aligned_idx_1, self.aligned_idx_2, self.n_align = self.dijkstra(
            c_m, self.v_s_1, self.v_s_2
        )

        ## Compute aligned vectors
        self.aligned_vs1 = self.v_s_1.T[self.aligned_idx_1].T
        self.aligned_vs2 = self.v_s_2.T[self.aligned_idx_2].T

        ## Compute aligned fixations, using the convention that the fixation
        ## is at the beginning of the saccade vector
        self.aligned_s1s = self.s_1_s.T[self.aligned_idx_1].T
        self.aligned_s2s = self.s_2_s.T[self.aligned_idx_2].T

        results = dict()
        results.update(
            {
                "shape": self.shape_diff(),
                "position": self.position_diff(),
                "angle": self.angle_diff(),
                "length": self.length_diff(),
                "duration": self.duration_diff(),
            }
        )

        self.results = results

        if config["display_results"]:
            plot_simplification(self.s_1, self.s_1_s, self.s_2, self.s_2_s, id_1, id_2)

            self.comp_vis_pairs(id_1, id_2)

    def simplification(self, s_):
        """


        Parameters
        ----------
        s_ : TYPE
            DESCRIPTION.

        Returns
        -------
        s_ : TYPE
            DESCRIPTION.

        """

        process = True
        i = 0

        while process:
            s_n_ = self.amplitude_simplification(s_)
            s_n_ = self.direction_simplification(s_n_)

            if len(s_n_[0]) == len(s_[0]):
                process = False

            if i >= self.m_i:
                process = False

            s_ = copy.deepcopy(s_n_)

        return s_

    def amplitude_simplification(self, s_):
        """


        Parameters
        ----------
        s_ : TYPE
            DESCRIPTION.

        Returns
        -------
        n_s_ : TYPE
            DESCRIPTION.

        """

        n = len(s_[0])

        ## Compute successive vectors
        d_v = np.zeros((2, n))
        d_v[:, 1:] = s_[:2, 1:] - s_[:2, :-1]

        ## Compute vector amplitudes
        a_s = np.linalg.norm(d_v, axis=0)

        ## Find amplitudes below a threshold value
        i_c_a = a_s <= self.amp_thrs
        i_c_a[0] = False

        ## Find fixation duration below a threshold value at the beginning of the saccade
        i_c_t_1 = s_[2, :-1] <= self.dur_thrs
        ## Insert a False at the beginning since indexes are shifted by 1 on the right
        i_c_t_1 = np.insert(i_c_t_1, 0, False)

        ## Find fixation duration below a threshold value at the end of the saccade
        i_c_t_2 = s_[2] <= self.dur_thrs

        ## Union of the intersection of the two duration conditions with the amplitude condition:
        ## True if under an amplitude threshold if the duration condition is fulfilled at the
        ## end or at the begining of the saccade
        i_c = np.array([a and b or a and c for a, b, c in zip(i_c_a, i_c_t_1, i_c_t_2)])

        ## Merge successive points leading to small amplitude vectors
        n_s_ = self.merging(i_c, s_)

        return n_s_

    def direction_simplification(self, s_):
        """


        Parameters
        ----------
        s_ : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        n = len(s_[0])

        ## Compute successive vectors
        d_v = np.zeros((2, n))
        d_v[:, 1:] = s_[:2, 1:] - s_[:2, :-1]

        ## Compute absolute angle between successive vectors
        ag_s = np.zeros(n)
        ag_s[1:-1] = np.array(
            [aad(d_v[:, i], d_v[:, i + 1]) for i in range(1, len(d_v[0]) - 1)]
        )
        ## Find angles below a threshold value
        i_c_d = ag_s <= self.ang_thrs

        i_c_d[0] = False
        i_c_d[-1] = False

        ## Find fixation duration below a threshold value
        i_c_t = s_[2] <= self.dur_thrs

        ## Intersection of the two conditions
        i_c = np.array([a and b for a, b in zip(i_c_d, i_c_t)])

        ## Merge successive points leading to similar direction vectors
        n_s_ = self.merging(i_c, s_)

        return n_s_

    def merging(self, i_c, s_):
        """


        Parameters
        ----------
        i_c : TYPE
            DESCRIPTION.
        s_ : TYPE
            DESCRIPTION.

        Returns
        -------
        n_s_ : TYPE
            DESCRIPTION.

        """

        n_s_ = copy.deepcopy(s_)
        t_c = np.where(i_c == True)[0]
        ## Initialize list of indexes to remove
        t_r = list()

        for k, g in groupby(enumerate(t_c), lambda ix: ix[0] - ix[1]):
            ## List of indexes to remove for this simplification sequence
            i_l = list(map(itemgetter(1), g))

            ## First index to keep
            idx_tk = i_l[0] - 1

            ## Sum duration of removed indexes
            n_s_[2, idx_tk] = np.sum(s_[2, idx_tk : i_l[-1] + 1])

            ## Add local list of indexes to remove to the global list
            t_r += i_l

        ## Remove data
        n_s_ = np.delete(n_s_, t_r, axis=1)

        return n_s_

    def dijkstra(self, c_m, s_1, s_2):
        """


        Parameters
        ----------
        c_m : TYPE
            DESCRIPTION.
        s_1 : TYPE
            DESCRIPTION.
        s_2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        n_1 = len(s_1[0])
        n_2 = len(s_2[0])

        ## Compute adjacency matrix
        a_m = np.zeros((n_1 * n_2, n_1 * n_2))

        for i in range(0, n_1 - 1):
            for j in range(0, n_2 - 1):
                a_m[i * n_2 + j, i * n_2 + j + 1] = c_m[i, j + 1]
                a_m[i * n_2 + j, (i + 1) * n_2 + j] = c_m[i + 1, j]
                a_m[i * n_2 + j, (i + 1) * n_2 + j + 1] = c_m[i + 1, j + 1]

        for i in range(0, n_1 - 1):
            a_m[i * n_2 + n_2 - 1, (i + 1) * n_2 + n_2 - 1] = c_m[i + 1, n_2 - 1]

        for j in range(0, n_2 - 1):
            a_m[(n_1 - 1) * n_2 + j, (n_1 - 1) * n_2 + j + 1] = c_m[n_1 - 1, j + 1]

        ## Compute Dijkstra path with Networkx, faster than Python implementation
        G = nx.from_numpy_array(a_m, create_using=nx.DiGraph())
        _opt_path = nx.dijkstra_path(G, 0, n_1 * n_2 - 1)

        ai_1, ai_2 = [], []

        for i, step in enumerate(_opt_path):
            u = step // n_2
            v = step % n_2

            ai_1.append(u)
            ai_2.append(v)

        return np.array(ai_1), np.array(ai_2), len(_opt_path)

    def shape_diff(self):
        """
        Calculate vector similarity of two scanpaths.

        Returns
        -------
        TYPE
            Array of vector differences between pairs of saccades of two scanpaths.

        """

        a_v1, a_v2 = self.aligned_vs1, self.aligned_vs2

        v_diff = np.linalg.norm(a_v1 - a_v2, axis=0)
        v_diff /= 2 * self.vf_diag

        return np.mean(v_diff)

    def position_diff(self):
        """
        Calculate position similarity of two scanpaths.

        Returns
        -------
        TYPE
            Array of position differences between pairs of saccades of two scanpaths.

        """

        a_s1, a_s2 = self.aligned_s1s, self.aligned_s2s

        p_diff = np.linalg.norm(a_s1[0:2] - a_s2[0:2], axis=0)
        p_diff /= self.vf_diag

        return np.mean(p_diff)

    def angle_diff(self):
        """
        Calculate direction similarity of two scanpaths.

        Returns
        -------
        TYPE
            Array of vector differences between pairs of saccades of two scanpaths.

        """

        a_v1, a_v2 = self.aligned_vs1, self.aligned_vs2

        ag_diff = np.array([aad(a_v1[:, i], a_v2[:, i]) for i in range(len(a_v1[0]))])
        ag_diff /= 180

        return np.mean(ag_diff)

    def length_diff(self):
        """
        Calculate length similarity of two scanpaths.

        Returns
        -------
        TYPE
            Array of length difference between pairs of saccades of two scanpaths.

        """

        a_v1, a_v2 = self.aligned_vs1, self.aligned_vs2

        le_diff = np.abs(np.linalg.norm(a_v1, axis=0) - np.linalg.norm(a_v2, axis=0))
        le_diff /= self.vf_diag

        return np.mean(le_diff)

    def duration_diff(self):
        """
        Calculate similarity of two scanpaths fixation durations.

        Returns
        -------
        TYPE
            Array of fixation duration differences between pairs of saccades from
            two scanpaths.

        """

        n_a = self.n_align
        a_s1, a_s2 = self.aligned_s1s, self.aligned_s2s

        norm_ = np.max((a_s1[2].reshape(1, n_a), a_s2[2].reshape(1, n_a)), axis=0)[0]

        du_diff = np.abs(a_s1[2] - a_s2[2])
        du_diff /= norm_

        return np.mean(du_diff)

    def comp_vis_pairs(self, id_1, id_2):
        a_s1 = self.aligned_s1s[0:2]
        a_s2 = self.aligned_s2s[0:2]

        a_i1 = self.aligned_idx_1
        a_i2 = self.aligned_idx_2

        n_a = self.n_align
        o_l = []

        for i in range(n_a):
            i_1 = a_i1[i]
            i_2 = a_i2[i]

            o_l.append([a_s1[:, i], a_s2[:, i], [i_1, i_2]])

        o_l = np.array(o_l)

        plot_multi_match(self.s_1_s, self.s_2_s, o_l, id_1, id_2)


class MultiMatchAlignment:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)

        if verbose:
            print("Processing MultiMatch Alignment...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        else:
            raise ValueError(
                "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"
            )

        self.config = scanpaths[0].config
        self.config.update({"verbose": verbose})
        self.config.update({"display_results": display_results})

        ## Keep max x and y sizes for gaze visual field
        x_size_max = np.max(
            np.array([scanpath.config["size_plan_x"] for scanpath in scanpaths])
        )
        y_size_max = np.max(
            np.array([scanpath.config["size_plan_y"] for scanpath in scanpaths])
        )

        vf_diag = np.linalg.norm(
            np.array([self.config["size_plan_x"], self.config["size_plan_y"]])
        )

        self.config.update(
            {
                "size_plan_x": x_size_max,
                "size_plan_y": y_size_max,
                "screen_diagonal": vf_diag,
            }
        )

        if "nb_samples" in self.config.keys():
            del self.config["nb_samples"]

        self.config.update(
            {
                "multimatch_simplification_iterations": kwargs.get(
                    "multimatch_simplification_iterations", 10
                ),
                "multimatch_simplification_amplitude_threshold": kwargs.get(
                    "multimatch_simplification_amplitude_threshold", 0.055 * vf_diag
                ),
                "multimatch_simplification_duration_threshold": kwargs.get(
                    "multimatch_simplification_duration_threshold", 0.400
                ),
                "multimatch_simplification_angular_threshold": kwargs.get(
                    "multimatch_simplification_angular_threshold", 35
                ),
            }
        )

        scanpaths = scanpaths
        n_sp = len(scanpaths)

        features = ["shape", "position", "angle", "length", "duration"]
        d_ms = dict()

        for feat_ in features:
            d_ms.update({feat_: np.zeros((n_sp, n_sp))})

        for j in range(1, n_sp):
            for i in range(j):
                e_a = MultiMatch(
                    [scanpaths[i], scanpaths[j]], self.config, id_1=str(i), id_2=str(j)
                )

                results = e_a.results

                for feat_ in features:
                    d_m = d_ms[feat_]
                    d_m[i, j] = d_m[j, i] = results[feat_]
                    d_ms.update({feat_: d_m})

        self.results = dict({"multimatch_metrics": d_ms})

        self.verbose()

        if verbose:
            print("...MultiMatch Alignment done\n")

    def verbose(self, add_=None):
        if self.config["verbose"]:
            print("\n --- Config used: ---\n")

            for it in self.config.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.config[it]
                    )
                )

            if add_ is not None:
                for it in add_.keys():
                    print(
                        "# {it}:{esp}{val}".format(
                            it=it, esp=" " * (50 - len(it)), val=add_[it]
                        )
                    )
            print("\n")


def scanpath_multimatch_alignment(input, **kwargs):
    mm = MultiMatchAlignment(input, **kwargs)
    results = mm.results

    return results
