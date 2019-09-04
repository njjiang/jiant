# commitbank (CB only)
for i in `seq 0 24`
do
   ./run_k_cls.sh .../bert-large-cased/commitbank  ../bert-large-cased/commitbank/model_state_pretrain_val_5.best.th edges-cb-environment $i
done
for i in `seq 0 24`
do
   ./run_k_cls.sh .../bert-large-cased/commitbank  ../bert-large-cased/commitbank/model_state_pretrain_val_5.best.th edges-cb-factive $i
done

# commitplus (MNLI only)
for i in `seq 0 24`
do
   ./run_k_cls.sh ../bert-large-cased/commitbank_plus  ../bert-large-cased/commitbank_plus/model_state_pretrain_val_87.best.th edges-cb-environment $i
done

for i in `seq 0 24`
do
   ./run_k_cls.sh ../bert-large-cased/commitbank_plus  ../bert-large-cased/commitbank_plus/model_state_pretrain_val_87.best.th edges-cb-factive $i
done

#commitplus (CB+MNLI)
for i in `seq 0 24`
do
    ./run_k_cls.sh ./bert-large-cased/commitbank_plus ./bert-large-cased/commitbank_plus/commitbank/model_state_target_train_val_2.best.th edges-cb-environment $i edgeprobe-plus_commitbank_plus
done
for i in `seq 0 24`
do
    ./run_k_cls.sh ../bert-large-cased/commitbank_plus ../bert-large-cased/commitbank_plus/commitbank/model_state_target_train_val_2.best.th edges-cb-factive $i edgeprobe-plus_commitbank_plus
done
