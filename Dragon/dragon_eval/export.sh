#!/usr/bin/env bash

./build.sh

docker save joeranbosma/dragon_eval:latest | gzip -c > dragon_eval.tar.gz
