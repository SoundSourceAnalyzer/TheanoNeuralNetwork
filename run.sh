#!/usr/bin/env bash
set -euvx

curdir=`pwd`
sudo docker run -p 8888:8888 -v ${curdir}:/notebooks -it --rm pcej/keras-jupyter-yaafe       
