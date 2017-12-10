#!/bin/bash
# This script converts all notebooks into python files and stores them in the
# tests/ folder. This is useful for debugging. Note the some ipyhton calls need
# to be manually remove still
set -o errexit
set -o nounset
set -o pipefail

if [ ! -d "labs/" ];then
    echo "$0 should be called from the lxmls-toolkit root"
    exit
fi

if [ ! -d "tests/" ];then
    mkdir tests/
fi

#
for notebook in $(find labs/ -iname '*.ipynb' | grep -v .ipynb_checkpoints/);do
    output_dir=tests/$(dirname $(echo $notebook | sed 's@^labs/@@'))
    if [ ! -d $output_dir ];then
        mkdir -p $output_dir
    fi
    jupyter nbconvert --to script $notebook --output-dir $output_dir
done
