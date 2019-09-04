#!/bin/bash

# Script to generate cb edge probing datasets.
#
# Usage:
#  ./process_cb_edgeprobe.sh /path/to/glue_data
#

set -eux
JIANT_DATA_DIR=${1:-"$HOME/glue_data"}  # path to glue_data directory
OUTPUT_DIR="${JIANT_DATA_DIR}/edges"
mkdir -p $OUTPUT_DIR
HERE=$(dirname $0)

function preproc_task() {
    TASK_DIR=$1
    # Extract data labels.
    python ../get_edge_data_labels.py -o $TASK_DIR/labels.txt \
      -i $TASK_DIR/*.json -s
    # Retokenize for each tokenizer we need.
    #python ../retokenize_edge_data.py -t "MosesTokenizer" $TASK_DIR/*.json
    #python ../retokenize_edge_data.py -t "OpenAI.BPE"     $TASK_DIR/*.json
    #python ../retokenize_edge_data.py -t "bert-base-uncased"  $TASK_DIR/*.json
    python ../retokenize_edge_data.py -t "bert-large-cased" $TASK_DIR/*.json

    # Convert the original version to tfrecord.
    python ../convert_edge_data_to_tfrecord.py $TASK_DIR/*.json
}

# first convert to json files and write them in OUTPUT_DIR
$HERE/convert_cb_edgeprobe_to_json.py -o $OUTPUT_DIR
# run tokenizers on the json files
preproc_task $OUTPUT_DIR/cb_environment
preproc_task $OUTPUT_DIR/cb_factive

