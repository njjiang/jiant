#!/bin/bash

TARGET_DIR=$1

THIS_DIR=$(pwd)

set -e
if [ ! -d $TARGET_DIR ]; then
  mkdir $TARGET_DIR
fi

function fetch_data() {
  mkdir -p $TARGET_DIR/raw
  pushd $TARGET_DIR/raw

  # Univeral Dependencies 1.2 for English Web Treebank (ewt) source text.
  wget https://github.com/UniversalDependencies/UD_English/archive/r1.2.tar.gz
  mkdir ud
  tar -zxvf r1.2.tar.gz -C ud

  # UDS_IH2 annotations.
  wget http://decomp.io/projects/factuality/factuality_eng_udewt.tar.gz
  tar -xvzf factuality_eng_udewt.tar.gz

  popd
}

fetch_data

# Join UD with uds_ih2 annotations.
python $THIS_DIR/convert_uds_ih2_annot.py --src_dir $TARGET_DIR/raw -o $TARGET_DIR

