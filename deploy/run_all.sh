#!/bin/bash
# Deployment experiments reported in Section 7 (multi-process deployment).
set -u
cd "$(dirname "$0")"
R=../results/deploy
mkdir -p $R
python3 -u run_deploy.py $R/bt2 --block-time 2 --rounds 30            > $R/bt2.txt 2>&1
python3 -u run_deploy.py $R/bt2_crash --block-time 2 --rounds 30 --crash 3@11 > $R/bt2_crash.txt 2>&1
python3 -u run_deploy.py $R/bt1 --block-time 1 --rounds 10            > $R/bt1.txt 2>&1
python3 -u run_deploy.py $R/bt5 --block-time 5 --rounds 10            > $R/bt5.txt 2>&1
echo done > $R/ALL_DONE
