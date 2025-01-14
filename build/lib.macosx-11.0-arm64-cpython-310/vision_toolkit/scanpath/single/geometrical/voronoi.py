#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:34:35 2023

@author: quentinlaborde
"""

import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import gamma
from shapely.geometry import Polygon

from vision_toolkit.visualization.scanpath.single.geometrical import plot_voronoi_cells


class VoronoiCells:
    def __init__(self, scanpath):
        """


        Parameters
        ----------
        scanpath : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # Initialize parameters
        self.x_size = scanpath.config["size_plan_x"]
        self.y_size = scanpath.config["size_plan_y"]

        self.fixations = scanpath.values[0:2]
        self.areas, self.new_vertices = self.comp_voronoi_areas()

        skewness = self.comp_skewness()
        gamma = self.comp_gamma()

        self.results = dict(
            {
                "skewness": skewness,
                "gamma_parameter": gamma,
                "voronoi_areas": self.areas,
            }
        )

        if scanpath.config["display_results"]:
            plot_voronoi_cells(
                scanpath.values, self.new_vertices, scanpath.config["display_path"]
            )

    def comp_skewness(self):
        """


        Returns
        -------
        skw : TYPE
            DESCRIPTION.

        """

        areas = np.array(self.areas)

        mu = np.mean(areas)
        sigma = np.std(areas)

        skw = np.sum((areas - mu) ** 3)
        skw /= (len(areas) - 1) * sigma**3

        return skw

    def comp_gamma(self):
        """


        Returns
        -------
        fit_scale : TYPE
            DESCRIPTION.

        """

        areas = np.array(self.areas)
        areas /= np.mean(areas)

        fit_shape, fit_loc, fit_scale = gamma.fit(areas)

        return fit_scale

    def comp_voronoi_areas(self):
        """


        Parameters
        ----------
        scanpath : TYPE
            DESCRIPTION.
        x_size : TYPE
            DESCRIPTION.
        y_size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        pts = self.fixations.T
        x_size, y_size = self.x_size, self.y_size

        vor = Voronoi(pts)
        regions, vertices = self.voronoi_finite_polygons_2d(
            vor, radius=max(x_size, y_size) ** 2
        )

        coords = [[0.0, 0.0], [x_size, 0.0], [x_size, y_size], [0.0, y_size]]
        mask = Polygon(coords)

        new_vertices = []
        cell_areas = []

        for region in regions:
            polygon = vertices[region]
            c_polygon = Polygon(polygon).intersection(mask)

            cell_areas.append(c_polygon.area)
            poly = np.array(
                list(
                    zip(
                        c_polygon.boundary.coords.xy[0][:-1],
                        c_polygon.boundary.coords.xy[1][:-1],
                    )
                )
            )
            new_vertices.append(poly)

        return cell_areas, new_vertices

    def voronoi_finite_polygons_2d(self, vor, radius):
        """


        Parameters
        ----------
        vor : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)

        ## Construct a map containing all ridges for a given point
        all_ridges = {}

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        ## Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                ## finite region
                new_regions.append(vertices)
                continue

            ## reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1

                if v1 >= 0:
                    ## finite ridge: already in the region
                    continue

                ## Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            ## sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)

            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            ## finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)
