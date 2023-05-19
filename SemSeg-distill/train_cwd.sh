#!/bin/bash

CITYSCAPES=/mnt/dataset/CIRKD/data/Cityscapes
PASCALVOC=/mnt/dataset/CIRKD/data/PascalVOC
NORM=channel
DIVERGENCE=kl
KD_WEIGHT=0
CWD_WEIGHT=1
TEMPERATURE=1
CE=True
KD=False
CWD=True
ADV=False
EKD=False
FEAT=False
AKD=False
SRRL=False
MGD=False
DKD=False
VERSION=XDED
GPU=0

NAME=${KD}_${ADV}_${FEAT}_${NORM}_${DIVERGENCE}_${CWD_WEIGHT}_${TEMPERATURE}_${VERSION}

python main.py --kd $KD \
               --adv ${ADV} \
               --cwd ${CWD} \
               --cwd_feat ${FEAT} \
               --akd ${AKD} \
               --ekd ${EKD} \
               --srrl ${SRRL} \
               --mgd ${MGD} \
               --dkd ${DKD} \
               --temperature $TEMPERATURE \
               --norm_type $NORM \
               --divergence $DIVERGENCE \
               --lambda_cwd $CWD_WEIGHT \
               --lambda_kd $KD_WEIGHT \
               --data_dir $CITYSCAPES \
               --save_name ${NAME} \
               --gpu $GPU \
               --random_mirror \
               --random_scale \
               --ce ${CE}
               #--hp_search --verbose