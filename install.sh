#!/usr/bin/env bash

python -m ensurepip --upgrade
pip install -U pip setuptools wheel
#pip install -U 'spacy[cuda11x]'
pip install -r requirements.txt

python -m spacy download en_core_web_trf
