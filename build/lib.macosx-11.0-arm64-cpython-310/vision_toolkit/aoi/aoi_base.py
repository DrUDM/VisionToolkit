# -*- coding: utf-8 -*-

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

from vision_toolkit.aoi.identification_algorithms.I_AP import process_IAP
from vision_toolkit.aoi.identification_algorithms.I_DP import process_IDP
from vision_toolkit.aoi.identification_algorithms.I_DT import process_IDT
from vision_toolkit.aoi.identification_algorithms.I_KM import process_IKM
from vision_toolkit.aoi.identification_algorithms.I_MS import process_IMS
from vision_toolkit.aoi.identification_algorithms.predefined import process_predefined
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.utils.identification_utils import (collapse_AoI,
                                               temporal_binning_AoI)


class AoISequence(Scanpath):
    def __init__(self, input, gaze_df=None, ref_image=None, **kwargs):
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

        ## Used when generating AoI from multiple sequences
        if isinstance(input, dict):
            aoi_temporal_binning = kwargs.get("AoI_temporal_binning", False)
            input["config"].update({"AoI_temporal_binning": aoi_temporal_binning})
            if aoi_temporal_binning == True:
                input["config"].update(
                    {
                        "AoI_temporal_binning_length": kwargs.get(
                            "AoI_temporal_binning_length", 0.250
                        )
                    }
                )
            self.generate_from_dict(input, aoi_temporal_binning)

        ## Used when generating AoI from single sequence
        else:
            verbose = kwargs.get("verbose", True)
            if verbose:
                print("Processing AoI Sequence...\n")

            if isinstance(input, pd.DataFrame):
                super().__init__(input, gaze_df, ref_image, **kwargs)

            elif isinstance(input, str):
                super().__init__(input, gaze_df, ref_image, **kwargs)

            elif isinstance(input, BinarySegmentation):
                super().__init__(input, gaze_df, ref_image, **kwargs)

            elif isinstance(input, Scanpath):
                self.__dict__ = input.__dict__.copy()

            else:
                raise ValueError(
                    "Input must be a Scanpath, or a BinarySegmentation, or a DataFrame, or a csv"
                )

            aoi_method = kwargs.get("AoI_identification_method", "I_AP")

            ## If HMM AoI identification, call MarkovBasedAnalysis
            if aoi_method == "I_HMM":
                self.generate_from_hmm(input, kwargs)

            ## Else, use dedicated identification algorithms
            else:
                aoi_temporal_binning = kwargs.get("AoI_temporal_binning", False)

                if aoi_temporal_binning == True:
                    self.config.update(
                        {
                            "AoI_temporal_binning_length": kwargs.get(
                                "AoI_temporal_binning_length", 0.250
                            )
                        }
                    )

                self.config.update(
                    {
                        "verbose": verbose,
                        "display_AoI_identification": kwargs.get(
                            "display_AoI_identification", True
                        ),
                        "display_AoI_path": kwargs.get("display_AoI_path", None),
                        "AoI_identification_method": aoi_method,
                        "AoI_temporal_binning": aoi_temporal_binning,
                    }
                )
                vf_diag = np.linalg.norm(
                    np.array([self.config["size_plan_x"], self.config["size_plan_y"]])
                )

                if aoi_method == "I_KM":
                    self.config.update(
                        {
                            "AoI_IKM_cluster_number": kwargs.get(
                                "AoI_IKM_cluster_number", "search"
                            ),
                        }
                    )
                    if self.config["AoI_IKM_cluster_number"] == "search":
                        self.config.update(
                            {
                                "AoI_IKM_min_clusters": kwargs.get(
                                    "AoI_IKM_min_clusters", 2
                                ),
                                "AoI_IKM_max_clusters": kwargs.get(
                                    "AoI_IKM_max_clusters", 15
                                ),
                            }
                        )

                elif aoi_method == "I_DT":
                    self.config.update(
                        {
                            "AoI_IDT_density_threshold": kwargs.get(
                                "AoI_IDT_density_threshold", 0.05 * vf_diag
                            ),
                            "AoI_IDT_min_samples": kwargs.get("AoI_IDT_min_samples", 5),
                        }
                    )

                elif aoi_method == "I_DP":
                    self.config.update(
                        {
                            "AoI_IDP_gaussian_kernel_sd": kwargs.get(
                                "AoI_IDP_gaussian_kernel_sd",
                                0.1 * vf_diag,
                            ),
                            "AoI_IDP_centers": kwargs.get("AoI_IDP_centers", "mean"),
                        }
                    )

                elif aoi_method == "I_MS":
                    self.config.update(
                        {
                            "AoI_IMS_bandwidth": kwargs.get(
                                "AoI_IMS_bandwidth", 0.05 * vf_diag
                            ),
                        }
                    )

                elif aoi_method == "I_AP":
                    self.config.update(
                        {
                            "AoI_IAP_centers": kwargs.get("AoI_IAP_centers", "mean"),
                        }
                    )

                elif aoi_method == "predefined":
                    self.config.update(
                        {
                            "AoI_predefined_coordinates": kwargs.get(
                                "AoI_predefined_coordinates", None
                            ),
                            "AoI_predefined_all": kwargs.get(
                                "AoI_predefined_all", True
                            ),
                        }
                    )
                    assert (
                        self.config["AoI_predefined_coordinates"] is not None
                    ), "AoI coordinates must be specified"

                self.dict_methods_aoi = dict(
                    {
                        "I_KM": process_IKM,
                        "I_DT": process_IDT,
                        "I_DP": process_IDP,
                        "I_MS": process_IMS,
                        "I_AP": process_IAP,
                        "predefined": process_predefined,
                    }
                )
                self.ref_image = ref_image

                self.identification_results = None
                self.sequence = None
                self.durations = None
                self.centers = None
                self.nb_aoi = None
                self.fixation_analysis = None

                self.process()

            if verbose:
                print("...AoI Sequence done\n")

    def process(self):
        """


        Returns
        -------
        None.

        """

        if self.config["AoI_identification_method"] == None:
            return
        self.identification_results = self.dict_methods_aoi[
            self.config["AoI_identification_method"]
        ](self.values, self.config, self.ref_image)
        self.sequence = self.identification_results["AoI_sequence"]
        self.durations = self.identification_results["AoI_durations"]
        self.centers = self.identification_results["centers"]
        self.nb_aoi = len(self.centers.keys())

        self.verbose()

    def generate_from_dict(self, input, aoi_temporal_binning):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.identification_results = None

        seq_ = input["sequence"]
        seq_dur = input["durations"]
        config = input.get("config", dict({}))

        if aoi_temporal_binning == True:
            seq_, seq_dur = temporal_binning_AoI(seq_, seq_dur, config)
        if aoi_temporal_binning == "collapse":
            seq_, seq_dur = collapse_AoI(seq_, seq_dur)

        self.sequence = seq_
        self.durations = seq_dur

        self.centers = input.get("centers", None)
        self.nb_aoi = input["nb_aoi"]

        self.values = None
        self.config = config
        self.fixation_analysis = input.get("fixation_analysis", None)

    def generate_from_hmm(self, input, kwargs):
        from vision_toolkit.aoi.markov_based.markov_based import MarkovBasedAnalysis

        display_identification = kwargs.get("display_AoI_identification", True)

        if not isinstance(input, AoISequence):
            kwargs.update(
                {
                    "AoI_identification_method": "I_KM",
                    "display_AoI_identification": False,
                }
            )
        kwargs.update({"display_results": display_identification})
        markov_analysis = MarkovBasedAnalysis(input, **kwargs)
        HMM_nb_iters = kwargs.get("AoI_HMM_number_iterations", 10)
        HMM_AoI_instance = kwargs.get("AoI_HMM_return_AoISequence_instance", True)
        HMM_AoI_model = kwargs.get("AoI_HMM_return_model_instance", True)

        results = markov_analysis.AoI_HMM(
            HMM_nb_iters, HMM_AoI_instance, HMM_AoI_model, get_results=True
        )

        self.__dict__ = results["AoI_HMM_AoISequence_instance"].__dict__.copy()


class AoIMultipleSequences:
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
        display_results = kwargs.get("display_results", False)
        display_scanpath = kwargs.get("display_scanpath", False)
        aoi_temporal_binning = kwargs.get("AoI_temporal_binning", False)
        kwargs.update({"AoI_temporal_binning": False})

        if verbose:
            print("Processing AoI from multiple sequences...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], str): 
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], BinarySegmentation):
            scanpaths = [Scanpath.generate(input_, **kwargs) for input_ in input]

        elif isinstance(input[0], Scanpath):
            scanpaths = input

        config = scanpaths[0].config
        config.update(
            {
                "verbose": verbose,
                "display_results": display_results,
                "display_scanpath": display_scanpath,
            }
        )

        ## Keep max x and y sizes for gaze visual field
        x_size_max = np.max(
            np.array([scanpath.config["size_plan_x"] for scanpath in scanpaths])
        )
        y_size_max = np.max(
            np.array([scanpath.config["size_plan_y"] for scanpath in scanpaths])
        )
        config.update({"size_plan_x": x_size_max, "size_plan_y": y_size_max})

        values = np.empty((3, 0))
        bounds = dict()

        for i, scanpath in enumerate(scanpaths):
            b_ = values.shape[1]
            bounds[i] = [b_, b_ + scanpath.values.shape[1]]
            values = np.concatenate((values, scanpath.values), axis=1)
        dict_ = dict({"values": values, "config": config})

        c_scanpath = Scanpath(dict_)
        c_aoi_seq = AoISequence(c_scanpath, **kwargs)

        c_centers = c_aoi_seq.centers
        c_seq = c_aoi_seq.sequence
        c_durations = c_aoi_seq.durations

        nb_aoi = c_aoi_seq.nb_aoi

        self.config = c_aoi_seq.config
        self.config.update({"AoI_temporal_binning": aoi_temporal_binning})

        ## Iniialize dict for results
        self.aoi_sequences = []
        kwargs.update({"AoI_temporal_binning": aoi_temporal_binning})

        for i in bounds.keys():
            ## Compute dictionary to initialize each aoi sequence
            dict_ = dict(
                {
                    "sequence": c_seq[bounds[i][0] : bounds[i][1]],
                    "durations": c_durations[bounds[i][0] : bounds[i][1]],
                    "centers": c_centers,
                    "nb_aoi": nb_aoi,
                    "fixation_analysis": scanpaths[i].fixation_analysis,
                    "config": self.config,
                }
            )
            self.aoi_sequences.append(AoISequence(dict_, **kwargs))

        self.verbose()

        if verbose:
            print("...Processing AoI from multiple sequences done\n")

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


def AoI_sequences(input, **kwargs):
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

    if isinstance(input, list):
        if isinstance(input[0], Iterable):
            if (
                all(isinstance(elem, str) for elem in input)
                and not input[0][-3:] == "csv"
            ):
                print("HEEEEREEE")
                verbose = kwargs.get("verbose", True)

                if verbose:
                    print("Processing AoI sequence from list of strings...")

                nb_aoi = len(set(input))
                centers = kwargs.get("AoI_centers", None)
                durations = kwargs.get("AoI_durations", None)

                dict_ = dict(
                    {
                        "sequence": input,
                        "durations": durations,
                        "centers": centers,
                        "nb_aoi": nb_aoi,
                        "config": dict({}),
                    }
                )
                results = AoISequence(dict_, **kwargs)

                if verbose:
                    print("...Processing AoI sequence from list of strings done")

                return results

            elif (
                all(isinstance(elem, str) for elem in input[0])
                and not input[0][-3:] == "csv"
            ):
                verbose = kwargs.get("verbose", True)

                if verbose:
                    print("Processing AoI sequences from lists of strings...")

                nb_aoi = len(set(list(itertools.chain.from_iterable(input))))
                centers = kwargs.get("AoI_centers", None)
                durations = kwargs.get("AoI_durations", None)

                if durations is not None:
                    assert len(durations) == len(
                        input
                    ), "Durations must be provided for each sequence"

                results = []

                for i, sequence in enumerate(input):
                    if durations is not None:
                        l_durations = durations[i]
                    else:
                        l_durations = None

                    dict_ = dict(
                        {
                            "sequence": sequence,
                            "durations": l_durations,
                            "centers": centers,
                            "nb_aoi": nb_aoi,
                            "config": dict({}),
                        }
                    )
                    results.append(AoISequence(dict_, **kwargs))

                if verbose:
                    print("...Processing AoI sequences from lists of strings done")

                return results

            else:
                m_seqs = AoIMultipleSequences(input, **kwargs)
                results = m_seqs.aoi_sequences
                return results

        else:
            m_seqs = AoIMultipleSequences(input, **kwargs)
            results = m_seqs.aoi_sequences
            return results

    else:
        aoi_seq = AoISequence(input, **kwargs)
        return aoi_seq
