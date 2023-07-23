#!/bin/bash

mkdir -p data
cd data

wget "https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/CampusSeq1.tar.bz2" -O "CampusSeq1.tar.bz"
tar -xvf CampusSeq1.tar.bz
rm CampusSeq1.tar.bz
mv CampusSeq1/ Campus/