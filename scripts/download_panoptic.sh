#!/bin/bash

curPath=$(dirname "$0")

mkdir -p data
mkdir -p data/Panoptic
cd data/Panoptic

# download 10 training sequences and 4 test sequences
bash ../../scripts/getData.sh 160422_ultimatum1
bash ../../scripts/getData.sh 160906_pizza1
bash ../../scripts/getData.sh 160224_haggling1
bash ../../scripts/getData.sh 160226_haggling1
bash ../../scripts/getData.sh 161202_haggling1
bash ../../scripts/getData.sh 160422_haggling1
bash ../../scripts/getData.sh 160906_ian1
bash ../../scripts/getData.sh 160906_ian2
bash ../../scripts/getData.sh 160906_ian3
bash ../../scripts/getData.sh 160906_ian5
bash ../../scripts/getData.sh 160906_band1
bash ../../scripts/getData.sh 160906_band2
bash ../../scripts/getData.sh 160906_band3
bash ../../scripts/getData.sh 160906_band4

# extract the images from the videos
bash ../../scripts/extractAll.sh 160422_ultimatum1
bash ../../scripts/extractAll.sh 160906_pizza1
bash ../../scripts/extractAll.sh 160224_haggling1
bash ../../scripts/extractAll.sh 160226_haggling1
bash ../../scripts/extractAll.sh 161202_haggling1
bash ../../scripts/extractAll.sh 160422_haggling1
bash ../../scripts/extractAll.sh 160906_ian1
bash ../../scripts/extractAll.sh 160906_ian2
bash ../../scripts/extractAll.sh 160906_ian3
bash ../../scripts/extractAll.sh 160906_ian5
bash ../../scripts/extractAll.sh 160906_band1
bash ../../scripts/extractAll.sh 160906_band2
bash ../../scripts/extractAll.sh 160906_band3
bash ../../scripts/extractAll.sh 160906_band4