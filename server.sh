#!/usr/bin/env bash

PYTHONPATH="${PYTHONPATH}:/citron" ./bin/citron-server --model-path ./models/en_2021-11-15 --host 127.0.0.1
