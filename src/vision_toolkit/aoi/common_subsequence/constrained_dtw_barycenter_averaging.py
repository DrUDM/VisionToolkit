# -*- coding: utf-8 -*-

import copy
import itertools
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.aoi.pattern_mining.n_gram import NGram
from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.utils.binning import aoi_dict_dist_mat
from vision_toolkit.visualization.scanpath.similarity.distance_based.elastic import plot_DTW_frechet


class CDBA:
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
            print("Processing CDBA common subsequence...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.aoi_sequences = aoi_sequences
        self.centers = aoi_sequences[0].centers
        self.aoi_ = list(self.centers.keys())
        self.d_m, self.i_dict = aoi_dict_dist_mat(self.centers, normalize=False)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_CDBA_initialization_length": kwargs.get(
                    "AoI_CDBA_initialization_length", "min"
                ),
                "AoI_CDBA_initial_random_state": kwargs.get(
                    "AoI_CDBA_initial_random_state", 1
                ),
                "AoI_CDBA_maximum_iterations": kwargs.get(
                    "AoI_CDBA_maximum_iterations", 20
                ),
                "verbose": verbose,
            }
        )

        self.relaxed = None
        self.common_subsequence = self.process_CDBA()

        if verbose:
            print("...CDBA common subsequence done\n")
        self.verbose()

    def process_CDBA(self):
        ## Get input AoISequences
        aoi_sequences = self.aoi_sequences
        ## Keep only str sequences
        S_s = [aoi_sequences[i].sequence for i in range(len(aoi_sequences))]

        ## Find all 2-Grams from the input sequences
        bi_occ = [list(NGram(s_, 2).table.keys()) for s_ in S_s]
        bi_occ = sorted(set(list(itertools.chain.from_iterable(bi_occ))))

        ## Compute maximal number of occurences for each AoI
        counts_ = self.get_counts(S_s)

        ## Initiate the consensus sequence wrt AoI_CDBA_initialization_length parameter
        if self.config["AoI_CDBA_initialization_length"] == "min":
            l_m = np.min([len(s_) for s_ in S_s])

        elif self.config["AoI_CDBA_initialization_length"] == "max":
            l_m = np.max([len(s_) for s_ in S_s])

        else:
            raise ValueError(
                "'AoI_CDBA_initialization_length' must be set to 'min' or 'max'"
            )

        np.random.seed(self.config["AoI_CDBA_initial_random_state"])
        consensus = list(np.random.choice(list(self.centers.keys()), l_m))
        old_consensus = copy.deepcopy(consensus)

        max_iter = self.config["AoI_CDBA_maximum_iterations"]
        iter_ = 0

        while iter_ < max_iter:
            alignments = self.perform_alignments(consensus, S_s)
            consensus = self.update_consensus(alignments, bi_occ, counts_)
            iter_ += 1

            if consensus == old_consensus:
                break
            else:
                old_consensus = copy.deepcopy(consensus)

        assert (
            self.relaxed == False
        ), "Impossible to satisfy the two constraints. Try to modify AoI_CDBA_initial_random_state"

        return consensus

    def update_consensus(self, alignments, bi_occ, counts_):
        """


        Parameters
        ----------
        alignments : TYPE
            DESCRIPTION.
        bi_occ : TYPE
            DESCRIPTION.
        counts_ : TYPE
            DESCRIPTION.

        Returns
        -------
        consensus : TYPE
            DESCRIPTION.

        """

        aoi_ = self.aoi_
        i_dict = self.i_dict
        d_m = self.d_m

        current_counts_ = dict.fromkeys(counts_.keys(), 0)
        consensus = []

        for i in sorted(list(alignments.keys())):
            al_ = alignments[i]

            if i == 0:
                avail = aoi_
                ## Compute sum of distances to aligned AoI for each candidate consensus AoI
                d_t = [
                    np.sum([d_m[i_dict[aoi], i_dict[al]] for al in al_])
                    for aoi in avail
                ]
                ## Get optimal AoI
                opt_aoi = avail[np.argmin(d_t)]

            else:
                ## Compute available AoI whose count does not exceed the maximum count
                avail = [aoi for aoi in aoi_ if current_counts_[aoi] < counts_[aoi]]
                ## Compute available AoI based on 2-Grams computed from input sequences
                avail = [
                    aoi
                    for aoi in avail
                    if ("{last},{aoi},".format(last=consensus[-1], aoi=aoi) in bi_occ)
                ]
                ## If no available AoI, relax the second assumption
                if avail == []:
                    self.relaxed = True
                    avail = [aoi for aoi in aoi_ if current_counts_[aoi] < counts_[aoi]]
                else:
                    self.relaxed = False
                ## Compute sum of distances to aligned AoI for each candidate consensus AoI
                d_t = [
                    np.sum([d_m[i_dict[aoi], i_dict[al]] for al in al_])
                    for aoi in avail
                ]
                ## Get optimal AoI
                opt_aoi = avail[np.argmin(d_t)]

            consensus.append(opt_aoi)
            current_counts_[opt_aoi] = current_counts_[opt_aoi] + 1

        return consensus

    def perform_alignments(self, consensus, S_s):
        """


        Parameters
        ----------
        consensus : TYPE
            DESCRIPTION.
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        alignments : TYPE
            DESCRIPTION.

        """

        centers = self.centers
        ## Convert into an array of AoI center positions
        consensus_a = np.array([centers[consensus[i]] for i in range(len(consensus))])
        ## Initialize a dictionary to keep AoI aligned with each element from
        ## the consensus sequence
        alignments = dict.fromkeys(range(len(consensus)), [])

        for k in range(len(S_s)):
            s_ = np.array(S_s[k])
            s_a = np.array([centers[s_[i]] for i in range(len(s_))])
            d_m = cdist(consensus_a, s_a, metric="euclidean")
            opt_links, dist_ = c_comparison.DTW(consensus_a.T, s_a.T, d_m)
            for i in range(len(consensus)):
                ## Find indexes of alignments which involve consensus[i]
                idx = np.argwhere((opt_links[:, 2, 0]) == i)[:, 0]
                ## Find elements from s_ aligned with consensus[i]
                al_idx = opt_links[idx, 2, 1].astype(int)
                al_ = s_[al_idx]

                ## Update AoI aligned with each element from the consensus sequence
                ## Note that several element AoI from one sequence can be aligned
                ## with each element from the consensus sequence
                alignments[i] = alignments[i] + list(al_)

        return alignments

    def get_counts(self, S_s):
        """


        Parameters
        ----------
        S_s : TYPE
            DESCRIPTION.

        Returns
        -------
        counts_ : TYPE
            DESCRIPTION.

        """

        aoi_ = self.aoi_
        counts_ = dict()

        for aoi in aoi_:
            c_ = [s_.count(aoi) for s_ in S_s]
            counts_.update({aoi: max(c_)})

        return counts_

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


def AoI_CDBA(input, **kwargs):
    
    cdba = CDBA(input, **kwargs)
    results = dict({"AoI_CDBA_common_subsequence": cdba.common_subsequence})

    return results






