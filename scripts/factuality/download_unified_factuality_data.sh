#!/usr/bin/env bash

#git clone https://github.com/gabrielStanovsky/unified-factuality.git

pushd unified-factuality/src
./scripts/download_external_corpora.sh
./scripts/install_converter.sh
./scripts/convert_corpora.sh


