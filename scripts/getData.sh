#!/bin/bash

datasetName=${1}
numHDViews=5 # Specify the number of hd views you want to donwload. Up to 31

# Select wget or curl, with appropriate options
if command -v wget >/dev/null 2>&1; then 
	WGET="wget -c"
	mO="-O"
elif command -v curl >/dev/null 2>&1; then
	WGET="curl -C -" 
	mO="-o"
else
	echo "This script requires wget or curl to download files."
	echo "Aborting."
	exit 1;
fi

# Each sequence gets its own subdirectory
mkdir $datasetName		
cd $datasetName


#####################
# Download hd videos
#####################
mkdir -p hdVideos
panel=0
nodes=(3 6 12 13 23)
for ((c=0; c<$numHDViews; c++))
do
  fileName=$(printf "hdVideos/hd_%02d_%02d.mp4" ${panel} ${nodes[c]})
  echo $fileName;
  # Download and delete if the file is blank
	cmd=$(printf "$WGET $mO hdVideos/hd_%02d_%02d.mp4 http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/videos/hd_shared_crf20/hd_%02d_%02d.mp4 || rm -v $fileName" ${panel} ${nodes[c]} ${panel} ${nodes[c]})
	eval $cmd
done

# Download calibration data
$WGET $mO calibration_${datasetName}.json http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/calibration_${datasetName}.json || rm -v calibration_${datasetName}.json

# 3D Body Keypoint (Coco19 keypoint definition)
# Download 3D pose reconstruction results (by vga index, coco19 format)
if [ ! -f hdPose3d_stage1_coco19.tar ]; then
$WGET $mO hdPose3d_stage1_coco19.tar  http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/hdPose3d_stage1_coco19.tar || rm -v hdPose3d_stage1_coco19.tar 
fi