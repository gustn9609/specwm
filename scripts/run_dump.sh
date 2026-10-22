#!/bin/bash
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH
python dump_all.py 2>&1 | grep -vE 'Warning|warn|xFormers|Using cache|Downloading|%|permute'
