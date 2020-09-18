#!/bin/bash

OVERRIDES="exp_name = EXP_single_task_factuality"
OVERRIDES+=", run_name = single-all_but_cb"
OVERRIDES+=", pretrain_tasks = all_but_cb"
#OVERRIDES+=", target_tasks = \"CB-factuality-idk,CB-factuality-you-know,CB-factuality-idk-comma,CB-factuality-you-know-comma,CB-factuality-idk-before-target,CB-factuality-you-know-before-target,CB-factuality-idk-comma-before-target,CB-factuality-you-know-comma-before-target,beaver,CB-factuality-target-only,CB-NoEnv,CB-factuality522\""
OVERRIDES+=", target_tasks = \"CB-all,mv2-all\""

OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", use_classifier = all_but_cb"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", cuda = 0"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
##
OVERRIDES+=", input_module=bert-large-cased"

## LOAD MNLI CHECKPOIONT
OVERRIDES+=", random_seed=80"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=CB-all"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
