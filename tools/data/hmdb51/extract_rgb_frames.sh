#!/usr/bin/env bash

cd ../
python build_rawframes.py D:\Mxd\mmaction\mmaction2\hmdb51\video D:\Mxd\mmaction\mmaction2\hmdb51\rawframes --task rgb --level 2  --ext mp4
echo "Genearte raw frames (RGB only)"

cd hmdb51/
