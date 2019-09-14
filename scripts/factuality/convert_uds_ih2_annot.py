#!/usr/bin/env python

# Much of the code in this file is from Rachel Rudinger's unreleased code.

# This script combines the original UD-English files (.conll) and
# UD it-happened annotation files (.tsv) to create a new .conll
# file in the style of the Stanovsky et al. 2017 unified-facutality
# files. The veridicality labels are converted to a [-3,3] scale,
# following the UW and unified-factuality conventions. Note that
# this conversion is LOSSY. It is intended to replicate the
# structure of the unified-factuality dataset; it does not preserve
# all information from the original UD it-happened annotation
# files. However, those files (.tsv) are/will also be released.

# Description of transformation:
# * Token indices are switched from 1-based (UD) to 0-based (unified)
#
# Usage:
#   ./convert_uds_ih2_annot.py --src_dir <uds_ih2_temp_dir> \
#       -o /path/to/jiant/data/uds_ih2

import argparse
import logging as log
import os
import re
import sys
from collections import defaultdict
from tqdm import tqdm
import copy

import numpy as np
import pandas as pd


log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def load_ud_english(fpath):
    """Load a file from the UD English corpus

    Parameters
    ----------
    fpath : str
        Path to UD corpus file ending in .conllu
    """
    n = 1
    fname = os.path.split(fpath)[1]
    parses = defaultdict(list)
    sent_ids = []
    for l in open(fpath):
        ident = fname + ' ' + str(n)

        if re.match(r'^\d', l):
            l_split = l.strip().split()
            parses[ident].append(l_split)
        elif parses[ident]:
            sent_ids.append(ident)
            n += 1

    return sent_ids, parses


def load_annotations(fpath):
    """Load annotation data

    Parameters
    ----------
    fpath : str
        Path to annotation data
    """

    sep = '\t' if fpath.split('.')[-1] == 'tsv' else ','
    if fpath:
        annotations = pd.read_csv(fpath, sep=sep)
        annotations = annotations.rename(columns=lambda x: x.replace('.', '').lower())
        annotations = annotations.rename(columns={'sentenceid': 'treeid', 'predtoken': 'nodeid'})

        annotations['nodeid'] = annotations['nodeid'].astype(str)

        return [annotations]
    else:
        return []


def ud_anno2scalar(row_dict):
    conf = row_dict["confidence"]
    happened = row_dict["happened"]
    if conf.lower() == "na" or happened.lower() == "na":
        return '_'
    conf = float(conf)
    # conf = 0, 1, 2, 3, or 4
    if happened.lower() == "true":
        polarity = 1.0
    elif happened.lower() == "false":
        polarity = -1.0
    else:
        raise TypeError
    return round(polarity * (3.0 / 4.0) * conf, 2)


def merge_scores(tok_anno):
    # check unfiltered scores
    scores = [ud_anno2scalar(row) for row in tok_anno]
    if len(tok_anno) == 0:
        score = '_'
    elif len(tok_anno) == 1:
        score = scores[0]
    elif len(tok_anno) == 2:
        if all([x == '_' for x in scores]):
            score = '_'
        else:
            score = round(np.mean([s for s in scores if not s == '_']), 3)
    else:
        raise ValueError("Should not have more than two annotations per instance.")
    return score


def convert_uds_ih2_split(split: str, ud_eng_anno, args):
    save_file = os.path.join(args.output_dir, f"{split}.conll")
    ud_eng_anno_split = ud_eng_anno.loc[lambda x: x.split == split, :]

    ud_eng_split_file = os.path.join(args.src_dir, "ud", "UD_English-EWT-r1.2", f"en-ud-{split}.conllu")
    # data_split is default_dict, key: 'en-ud-train.conllu 4778', val: list of conll rows (lists)
    sent_ids, ud_eng_split = load_ud_english(ud_eng_split_file)
    # anno_split: pandas.core.frame.DataFrame

    num_raw_anno = 0
    num_raw_anno_filt = 0
    with open(save_file, 'w') as save_fp:
        for sent_id in tqdm(sent_ids):
            conll_lines = ud_eng_split[sent_id]
            conll_lines_new = copy.deepcopy(conll_lines)
            sent_anno = ud_eng_anno_split.loc[lambda df: df.treeid == sent_id, :]
            # is the tsv token ids 1-indexed? yes
            for line in conll_lines_new:
                tok_index = line[0]  # 1-based indexing
                # if there are no annotations for the token, then all labels are '_'
                tok_anno = sent_anno.loc[lambda df: df.nodeid == tok_index, :]
                tok_anno_filt = tok_anno.loc[lambda df: df.keep == True, :]
                tok_anno = [row for _, row in tok_anno.iterrows()]
                tok_anno_filt = [row for _, row in tok_anno_filt.iterrows()]
                num_raw_anno += len(tok_anno)
                num_raw_anno_filt += len(tok_anno_filt)

                score_unfiltered = merge_scores(tok_anno)
                score_filtered = merge_scores(tok_anno_filt)

                # Pred.Token in annotations is also 1-based
                line[0] = str(int(tok_index) - 1)  # convert to 0-based indexing a la unified-factuality
                line[6] = str(int(line[6]) - 1)  # convert to 0-based indexing for the index of the head, too
                if int(line[6]) == -1:
                    line[6] = line[0]  # Unified format has ROOT point to itself, not to 0 or -1 fake root index
                line.insert(2, score_filtered)
                line.insert(3, score_unfiltered)
                save_fp.write('\t'.join([str(t) for t in line]) + '\n')
            save_fp.write("\n")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to source data (SPR and UD1.2), as passed " "to get_spr_data.sh",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory, e.g. /path/to/edges/data/spr2",
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    log.info("Processing UDS_IH2 annotations...")
    ud_eng_anno_file = os.path.join(args.src_dir, "it-happened", "it-happened_eng_ud1.2_07092017.tsv")
    ud_eng_anno: pd.DataFrame = load_annotations(ud_eng_anno_file)[0]

    for split in ["train", "dev", "test"]:
        convert_uds_ih2_split(split, ud_eng_anno, args)
    log.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
