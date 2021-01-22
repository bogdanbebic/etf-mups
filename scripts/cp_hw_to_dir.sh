#!/bin/bash

dst_dir=$1

root_dir=$(realpath `git rev-parse --git-dir`/..)

cp `find $root_dir -name dz?z?.c` $dst_dir
cp `find $root_dir -name dz?z?.cu` $dst_dir
cp `find $root_dir -name MPS_DZ?_izvestaj_*.pdf` $dst_dir
cp `find $root_dir -name MPS_DZ?_izvestaj_*.docx` $dst_dir
