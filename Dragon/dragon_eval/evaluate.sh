#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

docker volume create dragon_eval-output-$VOLUME_SUFFIX

# run evaluation
# note: change `dragon_eval-output-$VOLUME_SUFFIX` to a local path
# to store the output on your machine
docker run --rm \
        -v $SCRIPTPATH/test-predictions/:/input/ \
        -v dragon_eval-output-$VOLUME_SUFFIX:/output/ \
        joeranbosma/dragon_eval --tasks 000 001 002 003 004 005 006 007

# print evaluation results
docker run --rm \
        -v dragon_eval-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/metrics.json | python -m json.tool

docker volume rm dragon_eval-output-$VOLUME_SUFFIX
