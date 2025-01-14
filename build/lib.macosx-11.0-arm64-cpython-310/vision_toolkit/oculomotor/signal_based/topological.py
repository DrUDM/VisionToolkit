# -*- coding: utf-8 -*-
import copy

import gudhi as gd
import numpy as np

from vision_toolkit.oculomotor.full_data.full_data_base import FullData
from vision_toolkit.visualization.topological import (
    plot_betti_curve,
    plot_persistence_barcode,
    plot_persistence_curve,
    plot_persistence_diagram,
    plot_persistence_entropy,
    plot_persistence_landscape)


class TopologicalAnalysis(FullData):
    def __init__(
        self,
        input_df,
        sampling_frequency,
        methods=[
            "betti_curve",
            "persistence_curve",
            "persistence_landscape",
            "persistence_entropy",
        ],
        **kwargs
    ):
        super().__init__(input_df, sampling_frequency, methods, **kwargs)

        self.config.update(
            {"persistence_type": kwargs.get("persistence_type", "time_embedded")}
        )

        self.config.update(
            {"persistence_min_length": kwargs.get("persistence_min_length", 5)}
        )
        self.config.update({"curve_nb_points": kwargs.get("curve_nb_points", 1000)})

        self.config.update(
            {
                "persistence_landscape_order": kwargs.get(
                    "persistence_landscape_order", 30
                )
            }
        )

        # Compute persistence diagram
        self.pds = self.pd_compute(self.config["persistence_type"])

    def pd_compute(self, _type):
        n_s = self.config["nb_samples"]

        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]

        pos = np.concatenate((x_a.reshape(1, n_s), y_a.reshape(1, n_s)), axis=0)

        pds = dict({})

        for k, _dir in enumerate(["x", "y"]):
            if _type == "time_embedded":
                cc = gd.CubicalComplex(
                    dimensions=[n_s], top_dimensional_cells=pos[k, :]
                )
                dgm = cc.persistence()

            pd = []
            m_l = self.config["persistence_min_length"]

            for _item in dgm:
                if (_item[1][1] - _item[1][0]) >= m_l:
                    pd.append(list(_item[1]))

            pds[_dir] = pd

            if self.config["persistence_display"]:
                plot_persistence_diagram(pd, _dir)
                plot_persistence_barcode(pd, _dir)

        return pds

    def bc_compute(self, pds_a):
        n_i = len(pds_a)
        n_p = self.config["curve_nb_points"]

        grid_x = np.linspace(
            min(pds_a[:, 0]) - 1, max(pds_a[:, 1][pds_a[:, 1] != np.inf]) + 1, n_p
        )

        l_grid_y = np.zeros((n_i, n_p))

        for i in range(n_i):
            l_grid_y[i] = np.where(grid_x < pds_a[i, 1], 1, 0) * np.where(
                grid_x >= pds_a[i, 0], 1, 0
            )

        grid_y = np.sum(l_grid_y, axis=0)

        return grid_x, grid_y

    def betti_curve(self, get_raw=False):
        _type = self.config["persistence_type"]

        pds = self.pds

        grid_x = dict({})
        grid_y = dict({})

        if _type == "time_embedded":
            for _dir in ["x", "y"]:
                pds_a = np.array(pds[_dir])
                grid_x[_dir], grid_y[_dir] = self.bc_compute(pds_a)

            if self.config["display"]:
                plot_betti_curve(grid_x, grid_y)

            results = dict(
                {
                    "alpha_x": grid_x["x"],
                    "alpha_y": grid_x["y"],
                    "betti_number_x": grid_y["x"],
                    "betti_number_y": grid_y["y"],
                }
            )

            if not get_raw:
                return

            else:
                return results

    def pc_compute(self, pds_a):
        n_i = len(pds_a)
        n_p = self.config["curve_nb_points"]

        max_p = max(pds_a[:, 1][pds_a[:, 1] != np.inf]) + 1
        pds_a[pds_a == np.inf] = max_p

        grid_x = np.linspace(min(pds_a[:, 0]) - 1, max_p, n_p)

        l_grid_y = np.zeros((n_i, n_p))

        for i in range(n_i):
            l_grid_y[i] = np.where(grid_x < pds_a[i, 1], 1, 0) * np.where(
                grid_x >= pds_a[i, 0], 1, 0
            )

        d_i = pds_a[:, 1] - pds_a[:, 0]
        l_grid_y *= d_i.reshape((n_i, 1))

        grid_y = np.sum(l_grid_y, axis=0)

        return grid_x, grid_y

    def persistence_curve(self, get_raw=False):
        _type = self.config["persistence_type"]

        pds = self.pds

        grid_x = dict({})
        grid_y = dict({})

        if _type == "time_embedded":
            for _dir in ["x", "y"]:
                pds_a = np.array(pds[_dir])
                grid_x[_dir], grid_y[_dir] = self.pc_compute(pds_a)

            if self.config["display"]:
                plot_persistence_curve(grid_x, grid_y)

            results = dict(
                {
                    "alpha_x": grid_x["x"],
                    "alpha_y": grid_x["y"],
                    "persistence_x": grid_y["x"],
                    "persistence_y": grid_y["y"],
                }
            )

            if not get_raw:
                return

            else:
                return results

    def lambda_function(self, a, b, x):
        n_s = len(x)
        s_a = np.concatenate(((a + x).reshape(1, n_s), (b - x).reshape(1, n_s)), axis=0)

        out = np.maximum(0, np.min(s_a, axis=0))

        return out

    def pl_compute(self, pds_a, _order):
        n_i = len(pds_a)
        n_p = self.config["curve_nb_points"]

        max_p = max(pds_a[:, 1][pds_a[:, 1] != np.inf]) + 1
        pds_a[pds_a == np.inf] = max_p

        grid_x = np.linspace(min(pds_a[:, 0]) - 1, max_p, n_p)

        l_grid_y = np.zeros((n_i, n_p))

        for i in range(n_i):
            l_grid_y[i] = self.lambda_function(pds_a[i, 0], pds_a[i, 1], grid_x)
        l_grid_y.sort(axis=0)
        grid_y = np.flip(l_grid_y, 0)

        return grid_x, grid_y

    def persistence_landscape(self, get_raw=False):
        _type = self.config["persistence_type"]

        pds = self.pds

        grid_x = dict({})
        grid_y = dict({})
        _orders = dict({})

        if _type == "time_embedded":
            for _dir in ["x", "y"]:
                pds_a = np.array(pds[_dir])
                n_i = len(pds_a)

                _order = min(n_i, self.config["persistence_landscape_order"])

                _orders.update({_dir: _order})

                grid_x[_dir], grid_y[_dir] = self.pl_compute(pds_a, _order)

            if self.config["display"]:
                plot_persistence_landscape(grid_x, grid_y, _orders)

                results = dict(
                    {
                        "alpha_x": grid_x["x"],
                        "alpha_y": grid_x["y"],
                        "persistence_landscapes_x": grid_y["x"],
                        "persistence_landscapes_y": grid_y["y"],
                    }
                )

            if not get_raw:
                return

            else:
                return results

    def pe_compute(self, pds_a):
        n_i = len(pds_a)
        n_p = self.config["curve_nb_points"]

        max_p = max(pds_a[:, 1][pds_a[:, 1] != np.inf]) + 1
        pds_a[pds_a == np.inf] = max_p

        grid_x = np.linspace(min(pds_a[:, 0]) - 1, max_p, n_p)

        l_grid_y = np.zeros((n_i, n_p))

        for i in range(n_i):
            l_grid_y[i] = np.where(grid_x < pds_a[i, 1], 1, 0) * np.where(
                grid_x >= pds_a[i, 0], 1, 0
            )

        d_i = pds_a[:, 1] - pds_a[:, 0]
        l_grid_y *= d_i.reshape((n_i, 1))

        norm_ = (np.sum(l_grid_y, axis=0)).reshape(1, n_p)
        norm_[norm_ == 0.0] = 1

        l_grid_y /= norm_

        l_grid_0 = copy.deepcopy(l_grid_y)
        l_grid_0[l_grid_0 == 0.0] = 1

        grid_y = -1 * np.sum(l_grid_y * np.log10(l_grid_0), axis=0)

        return grid_x, grid_y

    def persistence_entropy(self, get_raw=False):
        _type = self.config["persistence_type"]

        pds = self.pds

        grid_x = dict({})
        grid_y = dict({})

        if _type == "time_embedded":
            for _dir in ["x", "y"]:
                pds_a = np.array(pds[_dir])
                grid_x[_dir], grid_y[_dir] = self.pe_compute(pds_a)

            if self.config["display"]:
                plot_persistence_entropy(grid_x, grid_y)

            results = dict(
                {
                    "alpha_x": grid_x["x"],
                    "alpha_y": grid_x["y"],
                    "persistence_entropy_x": grid_y["x"],
                    "persistence_entropy_y": grid_y["y"],
                }
            )

            if not get_raw:
                return

            else:
                return results
