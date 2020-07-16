#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
NNODES_NUMBER=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NNODES_NUMBER --node_rank=$NODE_RANK\
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:7}
