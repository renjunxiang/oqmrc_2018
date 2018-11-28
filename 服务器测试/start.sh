#!/bin/bash
export LANG="en_US.UTF-8"
cd /search/work
python3 /search/work/data_cut_word.py
python3 /search/work/model_word.py
