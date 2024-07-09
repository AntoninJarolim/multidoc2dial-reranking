#!/bin/bash

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$(pwd) #assuming you are in root repository folder

MONGODBSERVER=pcknot6.fit.vutbr.cz
DB_KEY=ce

if [ ! -d "data/naver_trecdl22-crossencoder-debertav3" ]; then
  cp -r ~/md2d_data/naver_trecdl22-crossencoder-debertav3 data/naver_trecdl22-crossencoder-debertav3
fi

hyperopt-mongo-worker --mongo=$MONGODBSERVER:1234/$DB_KEY --poll-interval=3
