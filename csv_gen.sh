#!/bin/bash

# Create train.csv
find data/train -type f -regex ".*v[vh].png" -exec basename -s .png {} \; | sed -En -e'1i Place,Datetime,x,y,type' -e's@_@,@g' -e's@[xy]-@@gp' > data/train.csv

# Create val.csv
find data/val -type f -regex ".*v[vh].png" -exec basename -s .png {} \; | sed -En -e'1i Place,Datetime,x,y,type' -e's@_@,@g' -e's@[xy]-@@gp' > data/val.csv
