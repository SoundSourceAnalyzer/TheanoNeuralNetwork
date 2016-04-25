#!/usr/bin/env bash
set -eu

sudo docker run -p 8888:8888 -v $(pwd):/notebooks -it --rm pcej/keras-jupyter-yaafe       
