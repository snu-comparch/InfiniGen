#!/bin/bash

for FIG in "figure11" "figure12" "figure13" "table2" "figure17"; do
cd $FIG
  sh run.sh
  cd ..
done
