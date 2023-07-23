#!/bin/bash
# Helper script to run other extraction tasks
# Input argument is output format for image files (png or jpg)

# Format for extracted images.
# Use png for best quality.
fmt=${2-jpg}

# Figure out the path of helper scripts
#DIR=$(dirname $(readlink -f $0))
COMMAND=$(perl -MCwd -e 'print Cwd::abs_path shift' $0)
DIR=$(dirname $COMMAND)
OLDDIR=$PWD

cd $1

# Extract 3D Keypoints
if [ -f hdPose3d_stage1_coco19.tar ]; then
	tar -xf hdPose3d_stage1_coco19.tar
fi


# Extract HD images
$DIR/hdImgsExtractor.sh ${fmt}

cd $OLDDIR