#!/bin/sh
echo $1

outputpng= echo $1 | sed s/dot/png/

echo $outputpng
dot -Tpng $1 > $outputpng
