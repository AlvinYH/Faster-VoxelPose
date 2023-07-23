#!/bin/bash

mkdir -p data
cd data/

wget "https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/Shelf.tar.bz2" -O "Shelf.tar.bz2"
tar -xvf Shelf.tar.bz2
rm Shelf.tar.bz2