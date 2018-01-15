#!/bin/sh

test=
# -z 判断变量是否为空   ===    ! -z 判断变量是否非空
if [ -z  $test ];then
    echo "2 \$test variable is zero!"
else
    echo "2 \$test variable is not zero!"
fi


if [ ! -z  $test ];then
    echo "\$test variable is not zero!"
else
    echo "\$test variable is zero!"
fi



   
