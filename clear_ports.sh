#!/bin/bash

# Clears local processes on ports in case of error

for x in `seq 2222 2232`
do
    echo $x
    echo `sudo lsof -t -i:$x`
done

for x in `seq 2222 2232`
do
    sudo kill `sudo lsof -t -i:$x`
done

