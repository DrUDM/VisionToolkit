from setuptools import Extension, setup

ext_modules = [
    Extension(
        "vision_toolkit.segmentation.segmentation_algorithms.c_I_DeT.c_I_DeT",
        sources=[
            "src/vision_toolkit/segmentation/segmentation_algorithms/c_I_DeT/c_I_DeT.pyx",
        ],
    ),
    Extension(
        "vision_toolkit.segmentation.segmentation_algorithms.c_I_HMM.c_I_HMM",
        sources=[
            "src/vision_toolkit/segmentation/segmentation_algorithms/c_I_HMM/c_I_HMM.pyx",
        ],
    ),
    Extension(
        "vision_toolkit.scanpath.similarity.c_comparison_algorithms.c_comparison_algorithms",
        sources=[
            "src/vision_toolkit/scanpath/similarity/c_comparison_algorithms/c_comparison_algorithms.pyx",
        ],
    ),
    Extension(
        "vision_toolkit.aoi.common_subsequence.local_alignment.c_alignment_algorithms.c_alignment_algorithms",
        sources=[
            "src/vision_toolkit/aoi/common_subsequence/local_alignment/c_alignment_algorithms/c_alignment_algorithms.pyx",
        ],
    ),
    Extension(
        "vision_toolkit.aoi.markov_based.c_HMM.c_HMM",
        sources=[
            "src/vision_toolkit/aoi/markov_based/c_HMM/c_HMM.pyx",
        ],
    ),
]


if __name__ == "__main__":
    import numpy as np
    from Cython.Build import cythonize

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
        include_dirs=[np.get_include()],
    )
