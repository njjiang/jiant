set -u

#EXP_DIR="./edgeprobe-commitbank"
#./probing/get_scalar_mix.py -i $EXP_DIR/probing-edges-cb-EmbeddingS/ $EXP_DIR/probing-edges-cb-factive/ $EXP_DIR/probing-edges-cb-MatSubjPer/ -o $EXP_DIR/scalars.tsv
#EXP_DIR="./edgeprobe-commitbank_plus"
#./probing/get_scalar_mix.py -i $EXP_DIR/probing-edges-cb-EmbeddingS/ $EXP_DIR/probing-edges-cb-factive/ $EXP_DIR/probing-edges-cb-MatSubjPer/ -o $EXP_DIR/scalars.tsv
#EXP_DIR="./edgeprobe-plus_commitbank_plus"
#./probing/get_scalar_mix.py -i $EXP_DIR/probing-edges-cb-EmbeddingS/ $EXP_DIR/probing-edges-cb-factive/ $EXP_DIR/probing-edges-cb-MatSubjPer/ -o $EXP_DIR/scalars.tsv

EXP_DIR="./edgeprobe-plus_commitbank_plus"
./probing/get_scalar_mix.py -i \
    $EXP_DIR/frozen-edges-cb-EmbeddingS-cls-mix-24/ \
    $EXP_DIR/frozen-edges-cb-factive-cls-mix-24/ \
    -o $EXP_DIR/frozen-scalars-emb-fact-cls-24.tsv

