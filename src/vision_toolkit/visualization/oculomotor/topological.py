# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_persistence_diagram(
    persistence, direction, alpha=0.6, band=0.0, inf_delta=0.1
):
    plt.style.use("seaborn-v0_8")

    # Sort by life time, then takes only the max_intervals elements
    persistence = sorted(
        persistence,
        key=lambda life_time: life_time[1] - life_time[0],
        # key=lambda life_time: life_time[1][1] - life_time[1][0],
        reverse=True,
    )

    (min_birth, max_death) = __min_birth_max_death(persistence, band)
    delta = (max_death - min_birth) * inf_delta

    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # line display of equation : birth = death
    x = np.linspace(axis_start, infinity, 1000)

    # infinity line and text
    plt.plot(x, x, color="k", linewidth=1.0)
    plt.plot(x, [infinity] * len(x), linewidth=1.0, color="k", alpha=alpha)
    plt.text(axis_start, infinity, r"$\infty$", color="k", alpha=alpha)

    # bootstrap band
    if band > 0.0:
        plt.fill_between(x, x, x + band, alpha=alpha, facecolor="red")

    # Draw points in loop
    for interval in reversed(persistence):
        if float(interval[1]) != float("inf"):
            # Finite death case
            plt.scatter(
                interval[0],
                interval[1],
                alpha=alpha,
                color=palette[direction],
            )
        else:
            # Infinite death case for diagram to be nicer
            plt.scatter(interval[0], infinity, alpha=alpha, color=palette[direction])

    if direction == "x":
        plt.title("Persistence diagram horizontal axis")

    elif direction == "y":
        plt.title("Persistence diagram vertical axis")

    plt.xlabel("Birth")
    plt.ylabel("Death")

    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])

    plt.show()
    plt.clf()


def __min_birth_max_death(persistence, band=0.0):
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][0]

    for interval in reversed(persistence):
        if float(interval[1]) != float("inf"):
            if float(interval[1]) > max_death:
                max_death = float(interval[1])

        if float(interval[0]) > max_death:
            max_death = float(interval[0])

        if float(interval[0]) < min_birth:
            min_birth = float(interval[0])

    if band > 0.0:
        max_death += band

    return (min_birth, max_death)


def plot_persistence_barcode(persistence, direction, alpha=0.6, inf_delta=0.1):
    # Sort by life time, then takes only the max_intervals elements
    persistence = sorted(
        persistence,
        key=lambda life_time: life_time[1] - life_time[0],
        reverse=True,
    )

    persistence = sorted(persistence, key=lambda birth: birth[0])

    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = (max_death - min_birth) * inf_delta

    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1]) != float("inf"):
            # Finite death case
            plt.barh(
                ind,
                (interval[1] - interval[0]),
                height=0.8,
                left=interval[0],
                alpha=alpha,
                color=palette[direction],
                linewidth=0,
            )
        else:
            # Infinite death case for diagram to be nicer
            plt.barh(
                ind,
                (infinity - interval[0]),
                height=0.8,
                left=interval[0],
                alpha=alpha,
                color=palette[direction],
                linewidth=0,
            )
        ind = ind + 1

    if direction == "x":
        plt.title("Persistence barcode horizontal axis")

    elif direction == "y":
        plt.title("Persistence barcode vertical axis")

    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, 0, ind])

    plt.show()
    plt.clf()


def plot_betti_curve(grid_x, grid_y, direction="both"):
    plt.style.use("seaborn-v0_8")

    if direction == "x":
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

    elif direction == "y":
        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    else:
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.xlabel("alpha", fontsize=9)
    plt.ylabel("betti number", fontsize=9)
    plt.legend(fontsize=7)

    plt.show()
    plt.clf()


def plot_persistence_curve(grid_x, grid_y, direction="both"):
    plt.style.use("seaborn-v0_8")

    if direction == "x":
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

    elif direction == "y":
        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    else:
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.xlabel("alpha", fontsize=9)
    plt.ylabel("persistence", fontsize=9)
    plt.legend(fontsize=7)

    plt.show()
    plt.clf()


def plot_persistence_landscape(grid_x, grid_y, _orders, direction="both"):
    if direction == "x":
        plt.style.use("seaborn-v0_8")

        for k in range(1, _orders["x"] + 1):
            plt.plot(
                grid_x["x"],
                grid_y["x"][k],
                linewidth=0.5,
                label="{k}-th order".format(k=k),
            )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("alpha", fontsize=9)
        plt.ylabel("persistence landscape", fontsize=9)
        plt.legend(fontsize=6)

        plt.show()
        plt.clf()

    elif direction == "y":
        plt.style.use("seaborn-v0_8")

        for k in range(1, _orders["y"] + 1):
            plt.plot(
                grid_x["y"],
                grid_y["y"][k],
                linewidth=0.5,
                label="{k}-th order".format(k=k),
            )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("alpha", fontsize=9)
        plt.ylabel("persistence landscape", fontsize=9)
        plt.legend(fontsize=6)

        plt.show()
        plt.clf()

    else:
        plt.style.use("seaborn-v0_8")

        for k in range(_orders["x"]):
            plt.plot(
                grid_x["x"],
                grid_y["x"][k],
                linewidth=0.5,
                label="{k}-th order".format(k=k + 1),
            )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("alpha", fontsize=9)
        plt.ylabel("persistence landscape", fontsize=9)
        plt.legend(fontsize=6)

        plt.show()
        plt.clf()

        for k in range(_orders["y"]):
            plt.plot(
                grid_x["y"],
                grid_y["y"][k],
                linewidth=0.5,
                label="{k}-th order".format(k=k + 1),
            )

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xlabel("alpha", fontsize=9)
        plt.ylabel("persistence landscape", fontsize=9)
        plt.legend(fontsize=6)

        plt.show()
        plt.clf()


def plot_persistence_entropy(grid_x, grid_y, direction="both"):
    plt.style.use("seaborn-v0_8")

    if direction == "x":
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

    elif direction == "y":
        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    else:
        plt.plot(
            grid_x["x"],
            grid_y["x"],
            linewidth=0.5,
            label="Horizontal axis",
            color="darkblue",
        )

        plt.plot(
            grid_x["y"],
            grid_y["y"],
            linewidth=0.5,
            label="Vertical axis",
            color="purple",
        )

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.xlabel("alpha", fontsize=9)
    plt.ylabel("persistence entropy", fontsize=9)
    plt.legend(fontsize=7)

    plt.show()
    plt.clf()


palette = {"x": "darkblue", "y": "purple"}
