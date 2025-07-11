#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

MEM_LIMIT="4g"

docker volume create dragon_eval-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test-predictions/:/input/ \
        -v dragon_eval-output-$VOLUME_SUFFIX:/output/ \
        joeranbosma/dragon_eval --tasks 101 102 103 104 105 106 107 108 109

docker run --rm \
        -v dragon_eval-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/metrics.json | python -m json.tool

docker run --rm \
        -v dragon_eval-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test-output-expected:/output-expected/ \
         python:3.10-slim python -c "import sys; import json; f1=json.load(open('/output/metrics.json')); f2=json.load(open('/output-expected/metrics.json')); sys.exit(int(f1!=f2));"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm dragon_eval-output-$VOLUME_SUFFIX
