import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm

from lib.utils.transforms import get_affine_transform, get_scale


ori_image_size_list = {
    'Panoptic': [1920, 1080],
    'Shelf': [1032, 776],
    'Campus': [360, 288],
}

image_size_list = {
    'Panoptic': [960, 512],
    'Shelf': [800, 608],
    'Campus': [800, 640],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--dataset', help='please specify the name of the dataset', 
                        choices=['Panoptic', 'Shelf', 'Campus'],
                        required=True, type=str)
    args, _ = parser.parse_known_args()
    return args


def get_resize_transform(ori_image_size, image_size):
    r = 0
    c = np.array([ori_image_size[0] / 2.0, ori_image_size[1] / 2.0])
    s = get_scale((ori_image_size[0], ori_image_size[1]), image_size)
    trans = get_affine_transform(c, s, r, image_size)
    return trans


def preprocess_panoptic(image_size, trans):
    data_dir = 'data/Panoptic'
    cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]

    TRAIN_LIST = [
        '160422_ultimatum1', '160224_haggling1', '160226_haggling1',
        '161202_haggling1', '160906_ian1', '160906_ian2',
        '160906_ian3', '160906_band1', '160906_band2',# '160906_band3',
    ]
    VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']

    train_interval = 3
    val_interval = 12
    
    # preprocess training data
    for seq in TRAIN_LIST:
        print("=> Start preprocessing the training sequence: {}".format(seq))
        anno_files = sorted(glob.glob(os.path.join(data_dir, seq, 'hdPose3d_stage1_coco19/*.json')))
        for i, anno_file in enumerate(tqdm(anno_files)):
            if i % train_interval != 0:
                continue
            for k in range(len(cam_list)):
                suffix = os.path.basename(anno_file).replace("body3DScene", "")
                prefix = "{:02d}_{:02d}".format(cam_list[k][0], cam_list[k][1])
                image_path = os.path.join(data_dir, seq, "hdImgs", prefix, prefix + suffix)
                image_path = image_path.replace("json", "jpg")

                image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                resized_image = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])),
                                            flags=cv2.INTER_LINEAR)
                cv2.imwrite(image_path, resized_image)

    # preprocess validation data
    for seq in VAL_LIST:
        print("=> Start preprocessing the validating sequence: {}".format(seq))
        anno_files = sorted(glob.glob(os.path.join(data_dir, seq, 'hdPose3d_stage1_coco19/*.json')))
        for i, anno_file in enumerate(tqdm(anno_files)):
            if i % val_interval != 0:
                continue
            for k in range(len(cam_list)):
                suffix = os.path.basename(anno_file).replace("body3DScene", "")
                prefix = "{:02d}_{:02d}".format(cam_list[k][0], cam_list[k][1])
                image_path = os.path.join(data_dir, seq, "hdImgs", prefix, prefix + suffix)
                image_path = image_path.replace("json", "jpg")

                image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                # resize the image
                if image.shape[0] != image_size[1] or image.shape[1] != image_size[0]:
                    resized_image = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])),
                                                flags=cv2.INTER_LINEAR)
                    cv2.imwrite(image_path, resized_image)


def preprocess_shelf(image_size, trans):
    data_dir = 'data/Shelf'
    frame_range = list(range(300, 601)) 
    num_views = 5

    print("=> Start preprocessing the Shelf dataset")
    for i in tqdm(frame_range):
        for k in range(num_views):
            image_path = os.path.join(data_dir, "Camera{}".format(k), "img_{:06d}.png".format(i))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            # resize the image
            if image.shape[0] != image_size[1] or image.shape[1] != image_size[0]:
                resized_image = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])),
                                            flags=cv2.INTER_LINEAR)
                cv2.imwrite(image_path, resized_image)


def preprocess_campus(image_size, trans):
    data_dir = 'data/Campus'
    frame_range = list(range(350, 471)) + list(range(650, 751))
    num_views = 3

    print("=> Start preprocessing the Campus dataset")
    for i in tqdm(frame_range):
        for k in range(num_views):
            image_path = os.path.join(data_dir, "Camera{}".format(k), "campus4-c{}-{:05d}.png".format(k, i))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            
            # resize the image
            if image.shape[0] != image_size[1] or image.shape[1] != image_size[0]:
                resized_image = cv2.warpAffine(image, trans, (int(image_size[0]), int(image_size[1])),
                                            flags=cv2.INTER_LINEAR)
                cv2.imwrite(image_path, resized_image)


if __name__ == '__main__':
    args = parse_args()

    # get resize transform
    ori_image_size = ori_image_size_list[args.dataset]
    image_size = image_size_list[args.dataset]
    trans = get_resize_transform(ori_image_size, image_size)

    if args.dataset == 'Panoptic':
        preprocess_panoptic(image_size, trans)
    elif args.dataset == 'Shelf':
        preprocess_shelf(image_size, trans)
    elif args.dataset == 'Campus':
        preprocess_campus(image_size, trans)