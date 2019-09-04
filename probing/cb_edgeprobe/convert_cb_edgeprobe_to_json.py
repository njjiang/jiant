#!/usr/bin/env python

# Script to convert DPR data into edge probing format.
#
# Usage:
#   ./convert_cb_edgeprobe.py -o /path/to/probing/data/dpr

import pandas as pd
import argparse
import json
import logging as log
import os
import sys

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)

def write_feature_probe_data(df, feature, output_dir):
    feat_out_dir = os.path.join(output_dir, f"cb_{feature}")
    if not os.path.exists(feat_out_dir):
        os.mkdir(feat_out_dir)
    for split, group in df.groupby("SuperGLUE_split"):
        out_f = open(os.path.join(feat_out_dir, f"cb_{feature}_{split}.json"), "w")
        for i, line in group.iterrows():
            out = dict()
            uid = line["uID"]
            out["info"] = {"uID": uid}
            out["targets"] = [{"span1": [0, 1], 
                               "label":  line[feature],
                               "span_text" : ""}]
            out["text"] = line["sentence1"]
            out_f.write(json.dumps(out))
            out_f.write("\n")
        out_f.close()
    log.info(f"Data written to {feat_out_dir}")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory, e.g. glue_data/edges/",
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    src_file = "cb_nli_feature.csv"
    df = pd.read_csv(src_file)
    log.info("Processing CB annotations...")
    for feat in ["factive", "environment"]:
        write_feature_probe_data(df, feat, args.output_dir)
    log.info("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
