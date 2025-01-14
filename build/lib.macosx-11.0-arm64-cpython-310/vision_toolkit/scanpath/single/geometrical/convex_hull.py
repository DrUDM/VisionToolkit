# -*- coding: utf-8 -*-
# Stores the result (points of convex hull)


import numpy as np

from vision_toolkit.visualization.scanpath.single.geometrical import plot_convex_hull

np.random.seed(1)


class ConvexHull:
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

        self.s_ = scanpath.values[0:2]
        self.n_ = len(scanpath.values[0])

        # Initialize hull
        self.h_ = set()
        self.h_a = None
        self.area = None

        self.comp_hull()
        self.results = dict({"hull_area": self.area, "hull_apex": self.h_a})

        if scanpath.config["display_results"]:
            plot_convex_hull(scanpath.values, self.h_a, scanpath.config["display_path"])

    def comp_hull(self):
        """


        Returns
        -------
        None.

        """

        s_ = self.s_.T
        n_ = self.n_

        assert n_ > 2, "Convex hull not possible"

        ## Finding the point with minimum and maximum x-coordinates
        s_x = np.argmax(s_[:, 0])
        i_x = np.argmin(s_[:, 0])

        ## Recursively find convex hull on each side
        self.quick_hull(s_[i_x], s_[s_x], 1)
        self.quick_hull(s_[i_x], s_[s_x], -1)

        h_a = []
        for p_ in self.h_:
            x = p_.split("$")
            l_p = [float(x[0]), float(x[1])]
            h_a.append(l_p)

        self.h_a = self.sort_coordinates(np.array(h_a))

        self.area = self.poly_area()

    ## End points of line L are p1 and p2. side can have value 1 or -1 specifying
    ## each of the parts made by the line L
    def quick_hull(self, p1, p2, side):
        s_ = self.s_.T
        n_ = self.n_

        ind = -1
        m_d = 0

        ## Finding the point with maximum distance from L and also on the specified side of L.
        for i in range(n_):
            l_d = self.line_dist(p1, p2, s_[i])

            if (self.find_side(p1, p2, s_[i]) == side) and (l_d > m_d):
                ind = i
                m_d = l_d

        ## If no point is found, add the end points of L to the convex hull.
        if ind == -1:
            self.h_.add("$".join(map(str, p1)))
            self.h_.add("$".join(map(str, p2)))

            return

        ## Recursion for the two parts divided by s_[ind]
        self.quick_hull(s_[ind], p1, -self.find_side(s_[ind], p1, p2))
        self.quick_hull(s_[ind], p2, -self.find_side(s_[ind], p2, p1))

    ## Returns the side of point p with respect to line joining points p1 and p2.
    def find_side(self, p1, p2, p):
        val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

        if val > 0:
            return 1

        if val < 0:
            return -1

        return 0

    ## Returns a value proportional to the distance between the point p and the
    ## line joining the points p1 and p2
    def line_dist(self, p1, p2, p):
        l_d = abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))

        return l_d

    ## Compute Shoelace formula for polygon area
    def poly_area(self):
        x_ = self.h_a[:, 0]
        y_ = self.h_a[:, 1]

        return 0.5 * np.abs(np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1)))

    ## Clockwise sorting of 2D coordinates
    def sort_coordinates(self, s_):
        x_m, y_m = s_.mean(0)
        x, y = s_.T

        a_s = np.arctan2(x - x_m, y - y_m)

        i_ = np.argsort(-a_s)

        return s_[i_]
