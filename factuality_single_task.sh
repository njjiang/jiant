set -e
TASK=${1:-"rp-random"}  

OVERRIDES="exp_name = EXP_single_task_factuality"
OVERRIDES+=", run_name = single-${TASK}"
OVERRIDES+=", pretrain_tasks = ${TASK}"
OVERRIDES+=", target_tasks = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", cuda = 0"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune" 
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=5, max_epochs = 20"
##
OVERRIDES+=", input_module=bert-large-cased"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB-factuality,CB-NoEnv,rp-random\""
OVERRIDES+=", use_classifier = ${TASK}"
OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", write_preds = \"val,test\""

pushd "${PWD%jiant*}jiant"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
