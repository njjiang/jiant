<<<<<<< HEAD
<<<<<<< HEAD:factuality_scripts/factuality_multitask.sh
#!/bin/bash
# Run multi-task factuality model
# Usage: ./factuality_scripts/factuality_multitask.sh
=======
# Run multi-task factuality model
# Usage: ./factuality_multitask.sh
>>>>>>> 2f8d37d... document scripts:factuality_scripts/factuality_multitask.sh
=======
#!/bin/bash
# Run multi-task factuality model
# Usage: ./factuality_scripts/factuality_multitask.sh
>>>>>>> 6de312e... add scripts and data
set -e
OVERRIDES="exp_name = EXP_multi_task_factuality"
OVERRIDES+=", run_name = \"3282020\""
OVERRIDES+=", pretrain_tasks = \"factbank,meantime,uw,uds-ih2,CB,CB2,rp\""
OVERRIDES+=", target_tasks = \"factbank,meantime,uw,uds-ih2,CB,CB2,rp\""
OVERRIDES+=", do_pretrain = 1"
OVERRIDES+=", do_target_task_training = 0"
OVERRIDES+=", do_full_eval = 1"
OVERRIDES+=", val_interval = 2000"
OVERRIDES+=", cuda = auto"
OVERRIDES+=", batch_size = 4"
OVERRIDES+=", write_preds = \"val,test\""
OVERRIDES+=", sent_enc = none, sep_embs_for_skip = 1, transfer_paradigm = finetune"
OVERRIDES+=", lr = .00001, min_lr = .0000001, lr_patience = 4, dropout=0.1, patience=5, max_epochs = 20"
OVERRIDES+=", input_module=bert-large-cased"

pushd "${PWD%jiant*}jiant"

python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"

OVERRIDES+=", do_pretrain = 0"
OVERRIDES+=", do_target_task_training = 1"
OVERRIDES+=", do_full_eval = 1"
python main.py -c jiant/config/defaults.conf -o "${OVERRIDES}"
