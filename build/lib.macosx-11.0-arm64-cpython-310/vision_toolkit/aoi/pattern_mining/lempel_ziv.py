# -*- coding: utf-8 -*-


from lempel_ziv_complexity import (lempel_ziv_complexity,
                                   lempel_ziv_decomposition)


class LemplZiv:
    def __init__(self, input, **kwargs):
        self.aoi = input

        lzc_, dec_ = self.process()
        self.results = dict(
            {"AoI_lempel_ziv_complexity": lzc_, "AoI_lempel_ziv_decomposition": dec_}
        )

    def process(self):
        seq = self.aoi.sequence
        seq = "".join(seq)
        lzc_ = lempel_ziv_complexity(seq)
        dec_ = lempel_ziv_decomposition(seq)

        return lzc_, dec_


def AoI_lempel_ziv(input, **kwargs):
    lz = LemplZiv(input, **kwargs)
    results = lz.results

    return results
