#!/bin/sh
echo $1

dot -Tpng $1 | display
