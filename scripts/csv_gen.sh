#!/bin/bash

set -e

script_dir=$(dirname $0)
output=

help_message="
USAGE
    ./$(basename $0) [OPTIONS] DATASET_DIR

    DATASET_DIR        Input filename

DESCRIPTION
    Generates CSV file with information of the images in the dataset

OPTIONS
    --output    Output filename.
                Default=DATASET_DIR.csv

EXAMPLE
    ./$(basename $0) --output output data/train
"

. $script_dir/parse_options.sh

dataset_dir=$1


if [[ -z $dataset_dir ]]; then
    echo "Illegal number of parameters"
    echo "$help_message"
    exit 1
fi

if [[ -z $output ]]; then
    output=$dataset_dir.csv
fi

# Create train.csv
find $dataset_dir -type f -name "*.png" | sed -En -e'h' \
    -e's@^.*/([[:alpha:]]+)_([[:digit:]]{8}t[[:digit:]]{6})/tiles/([[:alpha:]_]+)/.+_x-([[:digit:]]+)_y-([[:digit:]]+).*\.png$@\1,\2,\3,\4,\5@' \
    -e'G' -e 's@\n@,@gp' | sed '1i Place,Datetime,Type,x,y,Path' > $output
