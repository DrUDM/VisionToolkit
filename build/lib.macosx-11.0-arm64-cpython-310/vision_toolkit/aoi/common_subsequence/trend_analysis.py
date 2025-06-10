# -*- coding: utf-8 -*-

import copy
import operator
import re
from itertools import groupby

import numpy as np

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence

np.random.seed(15)


class TrendAnalysis:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing AoI String Edit Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_trend_analysis_tolerance_level": kwargs.get(
                    "AoI_trend_analysis_tolerance_level", 0.95
                ),
                "verbose": verbose,
            }
        )
        self.aoi_sequences = aoi_sequences

        self.common_subsequence = self.process_trend_analysis()
        self.verbose()

    def process_trend_analysis(self):
        """


        Returns
        -------
        trending_sequence : TYPE
            DESCRIPTION.

        """

        ## Initialize lists of AoI simplified sequences and concatenated AoI
        ## simplified sequence
        S_s, S_s_c = [], []
        ## Initialize lists of AoI simplified sequence durations and concatenated
        ## AoI simplified sequence durations
        D_s, D_s_c = [], []

        for i in range(len(self.aoi_sequences)):
            assert (
                self.aoi_sequences[i].durations is not None
            ), "AoI_durations must be provided to perform TrendAnalysis"

            s_s, d_s = self.simplify(
                self.aoi_sequences[i].sequence, self.aoi_sequences[i].durations
            )
            S_s.append(s_s)
            S_s_c += s_s
            D_s.append(d_s)
            D_s_c += d_s

        ## Get individual instances from the concatenated AoI sequence
        i_inst = sorted(list(set(S_s_c)))

        ## Compute list of shared instances, wrt the tolerance_level parameter
        shr_inst = self.get_shared(i_inst, S_s)

        ## Compute attentionnal thresholds from fully shared instances
        n_t, d_t = self.get_importance_thresholds(
            np.array(S_s_c), np.array(D_s_c), shr_inst
        )

        ## Compute list of other instance candidates
        c_inst = list(set(i_inst) - set(shr_inst))

        ## Remove unfrequent and short candidate instances
        c_inst = self.remove_candidates(
            c_inst, np.array(S_s_c), np.array(D_s_c), n_t, d_t
        )
        n_S_s = self.remove_instances(S_s, D_s, shr_inst, c_inst)
        trending_sequence = self.comp_trending_sequence(n_S_s, shr_inst, c_inst)

        return trending_sequence

    def comp_trending_sequence(self, n_S_s, shr_inst, c_inst):
        """


        Parameters
        ----------
        n_S_s : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.
        c_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        t_sp : TYPE
            DESCRIPTION.

        """

        i_inst = sorted(shr_inst + c_inst)

        ## Compute priority dictionnary
        priorities = {}
        for inst in i_inst:
            p_ = 0
            for n_s_s in n_S_s:
                pos_ = np.where(np.array(n_s_s) == inst)[0]
                if len(pos_) == 1:
                    p_ += 1 - (pos_[0] * (1 - 0.1) / (len(n_s_s) - 1))

            priorities.update({inst: p_})

        ## Compute threshold priority from shared instances
        p_t = min([priorities[s_inst] for s_inst in shr_inst])

        ## Sort priorities by decreasing order
        s_priorities = dict(
            sorted(priorities.items(), key=operator.itemgetter(1), reverse=True)
        )
        ## Initiate trending sequence and add instances by decreasing priority
        ## order
        t_sp = []
        for inst in s_priorities.keys():
            # Add all shared instances
            if inst in shr_inst:
                t_sp.append(re.split("(\d+)", inst)[0])

            # Add other instances if priority superior or equal to threshold
            ## priority
            else:
                if s_priorities[inst] >= p_t:
                    t_sp.append(re.split("(\d+)", inst)[0])

        ## Remove duplicates
        t_sp = [key for key, _group in groupby(t_sp)]
        return t_sp

    def remove_instances(self, S_s, D_s, shr_inst, c_inst):
        """


        Parameters
        ----------
        S_s : TYPE
            DESCRIPTION.
        D_s : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.
        c_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        n_S_s : TYPE
            DESCRIPTION.

        """

        ## Initiate new simplified AoI and duration sequence sets
        n_S_s = []

        for k, s_s in enumerate(S_s):
            ## Initiate new simplified AoI and duration sequence
            n_s_s = []
            ## Remove instances that are not shared or candidate
            for i in range(len(s_s)):
                if s_s[i] in shr_inst or s_s[i] in c_inst:
                    n_s_s.append(s_s[i])

            n_S_s.append(n_s_s)

        return n_S_s

    def remove_candidates(self, c_inst, S_s_c, D_s_c, n_t, d_t):
        """


        Parameters
        ----------
        c_inst : TYPE
            DESCRIPTION.
        S_s_c : TYPE
            DESCRIPTION.
        D_s_c : TYPE
            DESCRIPTION.
        n_t : TYPE
            DESCRIPTION.
        d_t : TYPE
            DESCRIPTION.

        Returns
        -------
        n_c_inst : TYPE
            DESCRIPTION.

        """

        ## Get candidates instances and initiate new candidate instance list
        c_inst = c_inst
        n_c_inst = []

        for inst in c_inst:
            idx = np.where(S_s_c == inst)[0]
            ## Get total number of occurences on the candidate instance
            if np.sum(D_s_c[idx, 1]) >= n_t:
                ## Get total duration of occurences on the candidate instance
                if np.sum(D_s_c[idx, 0]) >= d_t:
                    n_c_inst.append(inst)

        return n_c_inst

    def get_importance_thresholds(self, S_s_c, D_s_c, shr_inst):
        """


        Parameters
        ----------
        S_s_c : TYPE
            DESCRIPTION.
        D_s_c : TYPE
            DESCRIPTION.
        shr_inst : TYPE
            DESCRIPTION.

        Returns
        -------
        min_n : TYPE
            DESCRIPTION.
        min_d : TYPE
            DESCRIPTION.

        """

        ## Find instance indexes corresponding to shared element instances
        idx_c = [np.where(S_s_c == inst)[0] for inst in shr_inst]

        ## Compute the minimum total duration of the shared instances
        min_d = min([np.sum(D_s_c[idx, 0]) for idx in idx_c])

        ## Compute the minimum total number of occurrences for the shared instances
        min_n = min([np.sum(D_s_c[idx, 1]) for idx in idx_c])

        return min_n, min_d

    def get_shared(self, i_inst, S_s):
        """


        Parameters
        ----------
        i_inst : TYPE
            DESCRIPTION.
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        shr_inst : TYPE
            DESCRIPTION.

        """
        t_l = self.config["AoI_trend_analysis_tolerance_level"]
        shr_inst = []

        for inst in i_inst:
            ## Count the number of sequences with this instance
            in_ = 0
            for s_ in S_s:
                in_ += inst in s_

            if in_ / len(S_s) >= t_l:
                shr_inst.append(inst)

        return shr_inst

    ## Compute a simplified AoI sequence by collapsing consecutive repetitions
    ## and the corresponding duration vector
    def simplify(self, s_, d_):
        """


        Parameters
        ----------
        s_ : TYPE
            DESCRIPTION.
        d_ : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        s_s = [s_[0]]
        d_s = [[d_[0], 1]]

        for i in range(1, len(s_)):
            if s_[i] == s_[i - 1]:
                d_s[-1][0] += d_[i]
                d_s[-1][1] += 1
            else:
                s_s.append(s_[i])
                d_s.append([d_[i], 1])

        s_s = copy.deepcopy(np.array(s_s))
        d_s = np.array(d_s)

        n_s_s = np.empty_like(s_s, dtype="object")
        i_aoi = sorted(list(set(s_s)))

        ## Number visual elements by decreasing duration
        for aoi in i_aoi:
            idx = np.where(s_s == aoi)[0]

            ## Get indexes of decreasing order for duration values
            d_o_idx = [
                i[0]
                for i in sorted(
                    enumerate(d_s[idx, 0]), key=lambda k: k[1], reverse=True
                )
            ]

            for i in range(len(d_o_idx)):
                n_aoi = s_s[idx[i]] + str(i)
                n_s_s[idx[d_o_idx[i]]] = n_aoi

        return list(n_s_s), list(d_s)

    def verbose(self, add_=None):
        """


        Parameters
        ----------
        add_ : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

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


def AoI_trend_analysis(input, **kwargs):
    
    ta = TrendAnalysis(input, **kwargs)
    results = dict({"AoI_trend_analysis_common_subsequence": ta.common_subsequence})

    return results



