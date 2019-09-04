#!/bin/bash

# Script to run an edge-probing task on an existing trained model.
# Based on prob_example_run.sh

# Example for probing all 24 layer
# for i in `seq 0 24`
# do
#     ./run_k_cls.sh ../bert-large-cased/commitbank_plus ../bert-large-cased/commitbank_plus/commitbank/model_state_target_train_val_2.best.th edges-cb-factive $i
# done

# NOTE: don't be startled if you see a lot of warnings about missing parameters,
# like:
#    Parameter missing from checkpoint: edges-srl-conll2005_mdl.proj2.weight
# This is normal, because the probing task won't be in the original checkpoint.

MODEL_DIR=$1 # directory of checkpoint to probe,
             # e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PARAM_FILE=$2
PROBING_TASK=${3:-"edges-cb-environment"}  # probing task name(s)
echo $PARAM_FILE

MAX_LAYER=$4

EXP_NAME=${5:-"edgeprobe-$(basename $MODEL_DIR)"}  # experiment name
RUN_NAME="test-frozen-$PROBING_TASK-cls-mix-$MAX_LAYER" # name for this run
echo $EXP_NAME/$RUN_NAME

CONFIG_FILE=${MODEL_DIR}"/params.conf"
OVERRIDES="load_target_train_checkpoint = ${PARAM_FILE}"

OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", val_interval = 60"
OVERRIDES+=", pytorch_transformers_output_mode=mix"
OVERRIDES+=", pytorch_transformers_max_layer=${4}"
OVERRIDES+=", do_target_task_training=1"
OVERRIDES+=", do_full_eval=1"
OVERRIDES+=", write_preds=1"
OVERRIDES+=", reload_tasks=1"
OVERRIDES+=", reload_indexing=1"
OVERRIDES+=", reindex_tasks=$PROBING_TASK"
OVERRIDES+=", reload_vocab=1"

# copied from defaults.conf for edges-tmpl
OVERRIDES+=", span_classifier_loss_fn = \"sigmoid\""
OVERRIDES+=", classifier_span_pooling = \"attn\""
OVERRIDES+=", classifier_hid_dim = 256, classifier_dropout = 0.3, pair_attn = 0"

# save checkpoint for when max_layer is 24 -- using full BERT
if [ $MAX_LAYER -ne 24 ]; then
    OVERRIDES+=", delete_checkpoints_when_done=1"
fi
# `frozen` or `fine-tune` BERT layers 
OVERRIDES+=", transfer_paradigm=frozen"

pushd "${PWD%jiant*}jiant"

# Load defaults.conf for any missing params, then model param file,
# then eval_existing.conf to override paths & eval config.
# Finally, apply custom overrides defined above.
# To add email notifications, add an additional argument:
#   --notify my_email@example.com
python main.py -c jiant/config/defaults.conf ${CONFIG_FILE} jiant/config/edgeprobe/edgeprobe_existing.conf \
    -o "${OVERRIDES}" 
