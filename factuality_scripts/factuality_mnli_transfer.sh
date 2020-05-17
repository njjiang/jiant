# Run a single task model with MNLI transfer
# Usage: ./factuality_mnli_transfer.sh CB2 MNLI_CHECKPOINT_PATH

set -e
TASK=${1:-"CB"}
MNLI_CHECKPOINT=${2:-"bert-mnli/tuning-0/model_state_pretrain_val_76.best.th"}


OVERRIDES="exp_name = EXP_mnli_transfer_factuality"
OVERRIDES+=", run_name = transfer-to-${TASK}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
# but task specific val_interval will be loaded from defaults.conf
# OVERRIDES+=", val_interval = 100" 
OVERRIDES+=", cuda = 0"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=10, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=${TASK}"

## LOAD MNLI CHECKPOIONT
OVERRIDES+=", load_target_train_checkpoint = ${MNLI_CHECKPOINT}"
OVERRIDES+=", random_seed=81"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

# eval on all tasks
OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB,CB2,rp\""

OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", use_classifier = ${TASK}"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", reload_tasks=1, reload_indexing=1, reload_vocab=1, reindex_tasks=${TASK}"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

