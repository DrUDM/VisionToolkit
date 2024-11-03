# -*- coding: utf-8 -*-

import numpy as np


class TransitionEntropyAnalysis:
    def __init__(self, input):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.transition_matrix = input
        self.state_number = input.shape[0]
        self.stationary_distribution = self.compute_stationary_distribution()

        t_m_i = self.t_mutual_information()
        t_m_i_r = self.t_mutual_information_row()
        t_j_e = self.t_joint_entropy()
        t_c_e = self.t_conditional_entropy()
        t_c_e_r = self.t_conditional_entropy_row()
        t_s_e = self.t_stationary_entropy()

        self.results = dict(
            {
                "AoI_transition_stationary_entropy": t_s_e,
                "AoI_transition_joint_entropy": t_j_e,
                "AoI_transition_conditional_entropy": t_c_e,
                "AoI_transition_conditional_entropy_row": t_c_e_r,
                "AoI_transition_mutual_information": t_m_i,
                "AoI_transition_mutual_information_row": t_m_i_r,
            }
        )

    def compute_stationary_distribution(self):
        """


        Returns
        -------
        stationary_distrib : TYPE
            DESCRIPTION.

        """
        ## Find the stationary distribution as the left perron vector
        eigenvals, eigenvects = np.linalg.eig(self.transition_matrix.T)

        close_to_1_idx = np.isclose(eigenvals, 1)
        close_to_1_idx = self.find_nearest(eigenvals, 1)

        target_eigenvect = np.real(eigenvects[:, close_to_1_idx])
        ## Turn the eigenvector elements into probabilites
        stationary_distrib = target_eigenvect / sum(target_eigenvect)
        stationary_distrib = np.maximum(stationary_distrib, 0)
        stationary_distrib /= sum(stationary_distrib)

        return stationary_distrib

    def t_mutual_information(self):
        """


        Returns
        -------
        t_m_i : TYPE
            DESCRIPTION.

        """
        pi = self.stationary_distribution
        t_m = self.transition_matrix

        n_s = self.state_number

        t_m_i = 0
        for i in range(n_s):
            for j in range(n_s):
                if t_m[i, j] > 0 and pi[j] > 0:
                    t_m_i += pi[i] * t_m[i, j] * np.log(t_m[i, j] / pi[j])

        return np.real(t_m_i)

    def t_mutual_information_row(self):
        """


        Returns
        -------
        t_m_i : TYPE
            DESCRIPTION.

        """
        pi = self.stationary_distribution
        t_m = self.transition_matrix
        n_s = self.state_number

        t_m_i = dict()
        ## Compute the mutual information for each AoI
        for i in range(n_s):
            t_m_i_r = 0
            for j in range(n_s):
                if t_m[i, j] > 0 and pi[j] > 0:
                    t_m_i_r += t_m[i, j] * np.log(t_m[i, j] / pi[j])
            t_m_i.update({chr(i + 65): np.real(t_m_i_r)})

        return t_m_i

    def t_joint_entropy(self):
        """


        Returns
        -------
        t_j_e : TYPE
            DESCRIPTION.

        """
        pi = self.stationary_distribution
        t_m = self.transition_matrix
        n_s = self.state_number

        t_j_e = 0
        for i in range(n_s):
            for j in range(n_s):
                if (t_m[i, j] * pi[i]) > 0:
                    t_j_e += pi[i] * t_m[i, j] * np.log(pi[i] * t_m[i, j])

        return np.real(t_j_e)

    def t_conditional_entropy(self):
        """


        Returns
        -------
        t_c_e : TYPE
            DESCRIPTION.

        """
        pi = self.stationary_distribution
        t_m = self.transition_matrix
        n_s = self.state_number

        t_c_e = 0
        for i in range(n_s):
            t_c_e_r = 0
            for j in range(n_s):
                if t_m[i, j] > 0:
                    t_c_e_r -= t_m[i, j] * np.log(t_m[i, j])
            t_c_e += pi[i] * t_c_e_r

        return np.real(t_c_e)

    def t_conditional_entropy_row(self):
        """


        Returns
        -------
        t_c_e : TYPE
            DESCRIPTION.

        """
        t_m = self.transition_matrix
        n_s = self.state_number

        t_c_e = dict()
        for i in range(n_s):
            t_c_e_r = 0
            for j in range(n_s):
                if t_m[i, j] > 0:
                    t_c_e_r -= t_m[i, j] * np.log(t_m[i, j])
            t_c_e.update({chr(i + 65): t_c_e_r})

        return t_c_e

    def t_stationary_entropy(self):
        """


        Returns
        -------
        t_s_e : TYPE
            DESCRIPTION.

        """
        pi = self.stationary_distribution
        n_s = self.state_number

        t_s_e = 0
        for i in range(n_s):
            if pi[i] > 0:
                t_s_e -= pi[i] * np.log(pi[i])

        return np.real(t_s_e)

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx
