#!/bin/bash
true
export TORCH_HOME=$SCRATCH/.torch_hub; export HF_HUB_OFFLINE=1
cd $SCRATCH && python analysis_kvrw.py
