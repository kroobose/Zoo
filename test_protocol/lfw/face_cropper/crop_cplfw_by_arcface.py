"""
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import math
import multiprocessing

import cv2
sys.path.append('/workspace/face_sdk')
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

def crop_cplfw(cplfw_root, target_folder):
    face_cropper = FaceRecImageCropper()
    file_list = os.listdir(cplfw_root)
    for cur_file in file_list:
        if cur_file.endswith('.jpg'):
            cur_file_path = os.path.join(cplfw_root, cur_file)
            cur_image = cv2.imread(cur_file_path)

            face_lms = []
            cur_file_name = os.path.splitext(cur_file)[0]
            cur_lms_file_name = cur_file_name + '_5loc_attri.txt'
            cur_lms_file_path = os.path.join(cplfw_root, cur_lms_file_name)
            cur_lms_buf = open(cur_lms_file_path)
            line = cur_lms_buf.readline().strip()
            while line:
                line_strs = line.split(' ')
                face_lms.extend(line_strs)
                line = cur_lms_buf.readline().strip()
            face_lms = [float(s) for s in face_lms]
            face_lms = [int(num) for num in face_lms]
            cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, face_lms)
            target_path = os.path.join(target_folder, cur_file)
            cv2.imwrite(target_path, cur_cropped_image)

if __name__ == '__main__':
    cplfw_root = '/workspace/dataset/evaluation/CPLFW/images'
    target_folder = '/workspace/dataset/evaluation/CPLFW/cplfw_crop'
    crop_cplfw(cplfw_root, target_folder)
