#!/bin/bash
# Run single task models on a dataset
# Usage: ./factuality_single_task.sh CB2
# To run the Shared model, do: ./factuality_scripts/factuality_single_task.sh all-factuality

set -e
TASK=${1:-"rp"}

OVERRIDES="exp_name = EXP_single_task_factuality"
OVERRIDES+=", run_name = single-${TASK}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, dropout=0.1, max_epochs = 20"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=${TASK}"
##
OVERRIDES+=", input_module=bert-large-cased"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB,rp,mv2_2200\""
OVERRIDES+=", use_classifier = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", write_preds = \"val,test\""

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
