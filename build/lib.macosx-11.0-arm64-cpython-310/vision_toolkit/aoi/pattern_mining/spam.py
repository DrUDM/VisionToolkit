# -*- coding: utf-8 -*-

import copy
import sys
from bisect import bisect
from math import ceil
from random import randint
from typing import List

from bitarray import bitarray

from vision_toolkit.aoi.aoi_base import AoI_sequences, AoISequence
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class Prefix:
    def __init__(self, itemsets: List[List[int]] = None):
        self.itemsets: List[List[int]] = []
        if itemsets:
            self.itemsets.append(itemsets)

    def add_item_set(self, itemset: List[int]):
        self.itemsets.append(itemset)

    def clone_sequence(self):
        itemsets = copy.deepcopy(self.itemsets)
        prefix = Prefix()
        prefix.itemsets = itemsets
        return prefix

    def __len__(self):
        return len(self.itemsets)


class Bitmap:
    def __init__(self, last_bit_index):
        self.bitmap = None
        self.support = 0
        ## For calculating the support more efficiently
        ## we keep some information:
        ## the sid of the last sequence inserted in that bitmap that contains a bit set to 1
        self.last_sid = -1
        self.set_bit_array(last_bit_index + 1)

    def set_bit_array(self, last_bit_index):
        self.bitmap = bitarray(last_bit_index)
        self.bitmap.setall(0)

    def set_bit(self, index):
        ## print(f'Setting index: {index} on bitmap {self.bitmap} with length: {len(self.bitmap)}')
        self.bitmap[index] = True

    def register_bit(self, sid: int, tid: int, sequences_sizes: List[int]):
        """
        Determins right index in bitmap and sets bit to 1
        :param sid: sequence id
        :param tid: itemset id
        :param sequences_sizes: List[int] list of cumulative sequences sizes
        :return: None
        """
        # Get itemset index
        pos: int = sequences_sizes[sid] + tid
        self.bitmap[pos] = True

        if sid != self.last_sid:
            self.support += 1

        self.last_sid = sid

    def create_new_bitmap_s_step(
        self, bitmap: "Bitmap", sequences_size, last_bit_index
    ):
        new_bitmap = Bitmap(last_bit_index)
        ## get set bits on this bimap
        set_bits = self.bitmap.search(1)
        set_bits_candidate = bitmap.bitmap.search(1)
        ## must be only on first bits of every sequence
        first_bits = self.get_first_set_bits_of_every_sequence(
            set_bits, sequences_size, last_bit_index
        )
        for set_bit in first_bits:
            idx = bisect(set_bits_candidate, set_bit) - 1
            sid: int = self.bit_to_sid(set_bit, sequences_size)
            last_bit_of_sid: int = self.last_bit_of_sid(
                sid, sequences_size, last_bit_index
            )
            ## Get only sequence (SID) bits
            sequence_bits = [
                bit for bit in set_bits_candidate[idx + 1 :] if bit <= last_bit_of_sid
            ]
            match = False
            for next_bit in sequence_bits:
                new_bitmap.set_bit(next_bit)
                match = True

            if match:
                if sid != last_bit_of_sid:
                    new_bitmap.support += 1
        return new_bitmap

    def create_new_bitmap_i_step(
        self, bitmap: "Bitmap", sequences_size, last_bit_index
    ):
        new_bitmap = Bitmap(last_bit_index)
        set_bits = self.bitmap.search(1)
        # if both bits are TRUE
        for bit_idx in set_bits:
            if bitmap.bitmap[bit_idx]:
                new_bitmap.bitmap[bit_idx] = True

                sid: int = self.bit_to_sid(bit_idx, sequences_size)
                if sid != new_bitmap.last_sid:
                    new_bitmap.support += 1
                new_bitmap.last_sid = sid
        #  logical AND
        new_bitmap.bitmap = new_bitmap.bitmap & bitmap.bitmap

        return new_bitmap

    def bit_to_sid(self, bit, sequences_size):
        ## Bisect starts from index 1
        index = bisect(sequences_size, bit) - 1
        if index < 0:
            index = 0
        return index

    def last_bit_of_sid(self, sid, sequence_size, last_bit_index):
        if sid + 1 >= len(sequence_size):
            return last_bit_index
        return sequence_size[sid + 1] - 1

    def get_first_set_bits_of_every_sequence(
        self, set_bits, sequences_size, last_bit_index
    ):
        sid: int = -1
        first_bits = []
        for idx, bit in enumerate(set_bits):
            bit_sid = self.bit_to_sid(bit, sequences_size)
            if bit_sid != sid:
                first_bits.append(bit)
                sid = bit_sid
        return first_bits

    def __str__(self):
        return str(self.bitmap.tolist())


class SpamAlgo:
    def __init__(self, min_sup_rel):
        self.vertical_db = {}
        self.last_bit_index = None
        self.sequences_size = None
        ## relational minimal support
        self.min_sup_rel = min_sup_rel
        self.maximum_patter_length = sys.maxsize
        self.frequent_items = []

    def spam(self, sequences: List[List[List[int]]]):
        ## Go through all sequences and calculate sequences sizes
        self.last_bit_index, self.sequences_size = self.calculate_sequences_sizes(
            sequences
        )
        self.build_vertical_db(sequences)
        self.min_sup = self.calculate_min_support()
        self.remove_not_frequent_items()
        self.recursive_dfs()

    def calculate_sequences_sizes(self, sequences: List[List[List[int]]]):
        bit_index = 0
        sequences_sizes = [bit_index]
        for sequence in sequences:
            bit_index += len(sequence)
            if sequence != sequences[-1]:
                sequences_sizes.append(bit_index)
        last_bit_index = bit_index - 1
        return last_bit_index, sequences_sizes

    def build_vertical_db(self, sequences: List[List[List[int]]]):
        for sid, sequence in enumerate(sequences):
            for tid, itemset in enumerate(sequence):
                for idx, item in enumerate(itemset):
                    bitmap_item = self.vertical_db.get(item)
                    if not bitmap_item:
                        bitmap_item = Bitmap(self.last_bit_index)
                        self.vertical_db[item] = bitmap_item
                    bitmap_item.register_bit(sid, tid, self.sequences_size)

    def calculate_min_support(self):
        min_sup = ceil(self.min_sup_rel * len(self.sequences_size))
        if not min_sup:
            min_sup = 1

        return min_sup

    def remove_not_frequent_items(self):
        keys = list(self.vertical_db.keys())
        for k in keys:
            if self.vertical_db[k].support < self.min_sup:
                del self.vertical_db[k]
        self.frequent_items.extend([k] for k in self.vertical_db.keys())

    def recursive_dfs(self):
        keys = list(self.vertical_db.keys())
        for k in keys:
            prefix = Prefix([k])
            self.dfs_pruning(
                prefix=prefix,
                prefix_bitmap=self.vertical_db[k],
                search_s_items=list(self.vertical_db.keys()),
                search_i_items=list(self.vertical_db.keys()),
                has_to_be_greater_than_for_i_step=k,
                m=2,
            )

    def dfs_pruning(
        self,
        prefix: Prefix,
        prefix_bitmap: Bitmap,
        search_s_items: List[int],
        search_i_items: List[int],
        has_to_be_greater_than_for_i_step: int,
        m: int,
    ):
        s_temp, s_temp_bitmaps = self.perform_s_step(
            prefix, prefix_bitmap, search_s_items
        )
        for idx, item in enumerate(s_temp):
            prefix_s_step = prefix.clone_sequence()
            prefix_s_step.add_item_set([item])
            new_bitmap = s_temp_bitmaps[idx]

            self.frequent_items.append(prefix_s_step.itemsets)
            if self.maximum_patter_length > m:
                self.dfs_pruning(prefix_s_step, new_bitmap, s_temp, s_temp, item, m + 1)

        i_temp, i_temp_bitmaps = self.perform_i_step(
            prefix, prefix_bitmap, search_i_items, has_to_be_greater_than_for_i_step
        )
        for idx, item in enumerate(i_temp):
            ## create the new prefix
            prefix_i_step = prefix.clone_sequence()
            prefix_i_step.itemsets[len(prefix_i_step) - 1].append(item)
            ## create new Bitmap
            new_bitmap = i_temp_bitmaps[idx]
            if self.maximum_patter_length > m:
                self.dfs_pruning(prefix_i_step, new_bitmap, s_temp, i_temp, item, m + 1)

    def perform_s_step(self, prefix, prefix_bitmap, frequent_items):
        s_temp: List[int] = []
        s_temp_bitmaps: List[Bitmap] = []
        for i, k in enumerate(frequent_items):
            new_bitmap = prefix_bitmap.create_new_bitmap_s_step(
                bitmap=self.vertical_db[k],
                sequences_size=self.sequences_size,
                last_bit_index=self.last_bit_index,
            )

            if new_bitmap.support >= self.min_sup:
                s_temp.append(k)
                s_temp_bitmaps.append(new_bitmap)
        return s_temp, s_temp_bitmaps

    def perform_i_step(
        self,
        prefix,
        prefix_bitmap,
        frequent_items: List[int],
        has_to_be_greater_than_for_i_step: int,
    ):
        i_temp: List[int] = []
        i_temp_bitmaps: List[Bitmap] = []
        for item in frequent_items:
            if item > has_to_be_greater_than_for_i_step:
                new_bitmap = prefix_bitmap.create_new_bitmap_i_step(
                    self.vertical_db[item], self.sequences_size, self.last_bit_index
                )
                if new_bitmap.support >= self.min_sup:
                    i_temp.append(item)
                    i_temp_bitmaps.append(new_bitmap)
        return i_temp, i_temp_bitmaps


def generate_sequence():
    itemsets_in_sequence = randint(3, 10)
    sequence = [[] for _ in range(itemsets_in_sequence)]
    for i in range(itemsets_in_sequence):
        items_in_itemset = randint(1, 10)
        # generate items
        itemset = sequence[i]
        for j in range(items_in_itemset):
            item = randint(0, 20)
            if item not in itemset:
                itemset.append(item)
        itemset.sort()
    return sequence


# if __name__ == '__main__':
#     # sequence = generate_sequence()
#     # SpamAlgo().create_bitmap(sequence)
#     # l = [generate_sequence() for _ in range(4)]
#     sequences = [
#         [['a0', 'a1', 'a2'],['a2'],['a0']],
#         [['a0', 'a5'],['a2','a5']],
#         [['a0', 'a1'],['a1'],['a0'],['a5']]
#         ]
#     algo = SpamAlgo(0.5)
#     algo.spam(sequences)
#     print(algo.frequent_items)


class AoISPAM:
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
            print("Processing AoI String Edit Distance...\n")

        assert (
            len(input) > 1 and type(input) == list
        ), "Input must be a list of AoISequence, or a list of Scanpath, or a list of BinarySegmentation, or a list of csv"

        if isinstance(input[0], AoISequence):
            aoi_sequences = input

        else:
            aoi_sequences = AoI_sequences(input, **kwargs)

        self.config = aoi_sequences[0].config
        self.config.update(
            {
                "AoI_SPAM_support": kwargs.get("AoI_SPAM_support", 0.5),
                "verbose": verbose,
            }
        )
        self.aoi_sequences = aoi_sequences

        sequences = [
            [[s_] for s_ in aoi_sequence.sequence] for aoi_sequence in aoi_sequences
        ]

        algo = SpamAlgo(self.config["AoI_SPAM_support"])
        algo.spam(sequences)
        self.frequent_sequences = algo.frequent_items
        self.verbose()


def AoI_SPAM(input, **kwargs):
    sp_ = AoISPAM(input, **kwargs)
    results = dict({"AoI_trend_analysis_common_subsequence": sp_.frequent_sequences})

    return results
