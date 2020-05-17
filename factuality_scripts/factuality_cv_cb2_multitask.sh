# Train multi-task models on random splits of CB2 dataset
# Usage: for i in `seq 0 9`; do ./factuality_cv_cb2_multitask.sh $i; done

set -e
task="CB2"
split=${1:-"0"}
TASK="${task}-split${split}"

OVERRIDES="exp_name = EXP_multi_task_factuality"
OVERRIDES+=", run_name = \"${TASK}\""
OVERRIDES+=", pretrain_tasks = \"factbank,meantime,uw,uds-ih2,CB,$TASK,rp\""
OVERRIDES+=", target_tasks = \"${TASK}\""
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune"
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=5, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"
OVERRIDES+=", do_pretrain = 1"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 0"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", target_tasks = \"${TASK}\""
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", delete_checkpoints_when_done = 1"
python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
