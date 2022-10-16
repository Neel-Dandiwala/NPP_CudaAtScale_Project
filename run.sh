#!/usr/bin/env bash
make clean build

make run ARGS="-input=data/landscape.pgm"