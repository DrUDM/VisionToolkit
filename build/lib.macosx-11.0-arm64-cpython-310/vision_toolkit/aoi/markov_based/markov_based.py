# -*- coding: utf-8 -*-
import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoISequence
from vision_toolkit.aoi.markov_based.fisher_kernel import FisherKernel
from vision_toolkit.aoi.markov_based.hmm import AoIHMM
from vision_toolkit.aoi.markov_based.transition_entropy import TransitionEntropyAnalysis
from vision_toolkit.aoi.markov_based.transition_matrix import transition_matrix
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.markov_based.hmm import display_aoi_hmm
from vision_toolkit.visualization.aoi.transition_based.directed_graph import display_transition_matrix


class MarkovBasedAnalysis:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : str | BinarySegmentation | Scanpath | AoISequence
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)
       
        if verbose:
            print("Processing Markov Based Analysis...\n")

        if isinstance(input, str):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)
            

        elif isinstance(input, BinarySegmentation):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        elif isinstance(input, AoISequence):
            self.aoi_sequence = input

        elif isinstance(input, Scanpath):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        else:
            raise ValueError(
                "Input must be a csv, or a BinarySegmentation, or a Scanpath, or an AoISequence object"
            )

        self.transition_matrix = None

        if verbose:
            print("...Markov Based Analysis done\n")

    def AoI_transition_matrix(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        if self.transition_matrix is not None:
            results = dict({"AoI_transition_matrix": self.transition_matrix})
            if self.aoi_sequence.config["display_results"]:
                display_transition_matrix(self.transition_matrix)
            return results

        else:
            self.transition_matrix = transition_matrix(
                self.aoi_sequence.sequence, self.aoi_sequence.nb_aoi
            )
            results = dict({"AoI_transition_matrix": self.transition_matrix})
            if self.aoi_sequence.config["display_results"]:
                display_transition_matrix(self.transition_matrix)
            return results

    def AoI_transition_entropy(self):
        """


        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        if self.transition_matrix is None:
            t_m = self.AoI_transition_matrix()["AoI_transition_matrix"]
        else:
            t_m = self.transition_matrix
            if self.aoi_sequence.config["display_results"]:
                display_transition_matrix(t_m)

        e_a = TransitionEntropyAnalysis(t_m)
        results = e_a.results

        return results

    def AoI_HMM(self, HMM_nb_iters, HMM_AoI_instance, HMM_model, get_results):
        """


        Parameters
        ----------
        HMM_nb_iters : TYPE
            DESCRIPTION.
        HMM_AoI_instance : TYPE
            DESCRIPTION.
        get_results : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        print("Processing HMM AoI sequence identification...\n")

        centers_ = self.aoi_sequence.centers
        means_ = np.array([centers_[k_] for k_ in centers_.keys()])
        hmm = AoIHMM(
            None,
            means_,
            None,
            None,
            None,
            HMM_nb_iters,
        )
        hmm.infer_parameters(self.aoi_sequence.values[:2])

        aoi_seq = hmm.aoi_seq
        transition_mat = hmm.transition_matrix
        infered_centers = hmm.centers
        infered_covars = hmm.covars
        init_distrib = hmm.initial_distribution

        centers_ = dict({})
        clus_ = dict({})

        for i in range(transition_mat.shape[0]):
            vals_ = np.argwhere(np.array(aoi_seq) == int(i)).T[0]
            clus_.update({chr(i + 65): vals_})

            ## Centers are computed as the mean position of clustered fixations
            centers_.update({chr(i + 65): infered_centers[i]})

        ## Compute final AoI sequence
        seq_, seq_dur = compute_aoi_sequence(
            aoi_seq, self.aoi_sequence.values[2], self.aoi_sequence.config
        )

        if self.aoi_sequence.config["display_results"]:
            display_aoi_hmm(
                self.aoi_sequence.values[:2],
                clus_,
                infered_centers,
                infered_covars,
                self.aoi_sequence.config,
            )

        self.aoi_sequence.sequence = seq_
        self.aoi_sequence.centers = centers_
        self.aoi_sequence.durations = seq_dur
        self.aoi_sequence.identification_results = None

        self.aoi_sequence.config["AoI_identification_method"] = "I_HMM"
        self.transition_matrix = transition_mat

        if self.aoi_sequence.config["display_results"]:
            display_transition_matrix(transition_mat)

        for it_ in [
            "AoI_IKM_cluster_number",
            "AoI_IKM_min_clusters",
            "AoI_IKM_max_clusters",
        ]:
            if it_ in self.aoi_sequence.config.keys():
                del self.aoi_sequence.config[it_]

        self.aoi_sequence.verbose()

        print("...HMM AoI sequence identification done\n")

        if get_results:
            results = dict(
                {
                    "AoI_HMM_sequence": seq_,
                    "AoI_HMM_centers": infered_centers,
                    "AoI_HMM_covariances": infered_covars,
                    "AoI_HMM_transition_matrice": transition_mat,
                    "AoI_HMM_initial_distribution": init_distrib,
                }
            )

            if HMM_AoI_instance:
                AoI_seq = copy.deepcopy(self.aoi_sequence)
                results.update({"AoI_HMM_AoISequence_instance": AoI_seq})

            if HMM_model:
                results.update({"AoI_HMM_model_instance": hmm})

            return results


def AoI_transition_matrix(input, **kwargs):
    if isinstance(input, MarkovBasedAnalysis):
        results = input.AoI_transition_matrix()

    else:
        markov_analysis = MarkovBasedAnalysis(input, **kwargs)
        results = markov_analysis.AoI_transition_matrix()

    return results


def AoI_successor_representation(input, **kwargs):
    return 0


def AoI_transition_entropy(input, **kwargs):
    if isinstance(input, MarkovBasedAnalysis):
        results = input.AoI_transition_entropy()

    else:
        markov_analysis = MarkovBasedAnalysis(input, **kwargs)
        results = markov_analysis.AoI_transition_entropy()

    return results


def AoI_HMM(input, **kwargs):
    assert not isinstance(
        input, AoISequence
    ), "Input must be a csv, or a BinarySegmentation, or a Scanpath object"

    ## Initial centers are computed from K-Means using silhouette-score strategy
    kwargs.update(
        {"AoI_identification_method": "I_KM", "AoI_IKM_cluster_number": "search"}
    )
    markov_analysis = MarkovBasedAnalysis(input, **kwargs)
    HMM_nb_iters = kwargs.get("AoI_HMM_number_iterations", 10)
    HMM_AoI_instance = kwargs.get("AoI_HMM_return_AoISequence_instance", True)
    HMM_AoI_model = kwargs.get("AoI_HMM_return_model_instance", True)

    results = markov_analysis.AoI_HMM(
        HMM_nb_iters, HMM_AoI_instance, HMM_AoI_model, get_results=True
    )

    return results


def AoI_HMM_transition_matrix(input, **kwargs):
    assert not isinstance(
        input, AoISequence
    ), "Input must be a csv, or a BinarySegmentation, or a Scanpath object"

    ## Initial centers are computed from K-Means using silhouette-score strategy
    kwargs.update(
        {"AoI_identification_method": "I_KM", "AoI_IKM_cluster_number": "search"}
    )
    markov_analysis = MarkovBasedAnalysis(input, **kwargs)
    HMM_nb_iters = kwargs.get("AoI_HMM_number_iterations", 10)
    HMM_AoI_instance = kwargs.get("AoI_HMM_return_AoISequence_instance", True)

    markov_analysis.AoI_HMM(
        HMM_nb_iters, HMM_AoI_instance=False, HMM_model=False, get_results=False
    )
    results = markov_analysis.AoI_transition_matrix()

    return results


def AoI_HMM_transition_entropy(input, **kwargs):
    assert not isinstance(
        input, AoISequence
    ), "Input must be a csv, or a BinarySegmentation, or a Scanpath object"

    ## Initial centers are computed from K-Means using silhouette-score strategy
    kwargs.update(
        {"AoI_identification_method": "I_KM", "AoI_IKM_cluster_number": "search"}
    )
    markov_analysis = MarkovBasedAnalysis(input, **kwargs)
    HMM_nb_iters = kwargs.get("AoI_HMM_number_iterations", 10)
    HMM_AoI_instance = kwargs.get("AoI_HMM_return_AoISequence_instance", True)

    markov_analysis.AoI_HMM(
        HMM_nb_iters, HMM_AoI_instance=False, HMM_model=False, get_results=False
    )
    results = markov_analysis.AoI_transition_entropy()

    return results


def AoI_HMM_fisher_vector(input, **kwargs):
    HMM_model = kwargs.get("AoI_HMM_model", None)

    if HMM_model is None:
        centers = kwargs.get("AoI_HMM_centers", None)
        covars = kwargs.get("AoI_HMM_covariances", None)
        T = kwargs.get("AoI_HMM_transition_matrice", None)
        pi = kwargs.get("AoI_HMM_initial_distribution", None)

        assert None not in [
            centers,
            covars,
            T,
            pi,
        ], "Reference 'AoI_HMM_centers', 'AoI_HMM_covariances', 'AoI_HMM_transition_matrice', and 'AoI_HMM_initial_distribution' must be specified, or an 'AoI_HMM_model' instance must be provided"

        HMM_model = AoIHMM(None, centers, covars, pi, T, None)
    else:
        assert isinstance(
            HMM_model, AoIHMM
        ), "'AoI_HMM_model must be an AoIHMM instance"

    if isinstance(input, list):
        for i, observation in enumerate(input):
            print(observation)

    else:
        observation = input
        fk = FisherKernel(observation, HMM_model)
        fv = fk.fisher_vector
        results = dict(
            {"AoI_HMM_fisher_vector": fv, "AoI_HMM_model_instance": HMM_model}
        )

    return results
