#!/usr/bin/env python
import tensorflow as tf
import dataset_util

import glob
import argparse
import cv2
import csv

def get_index(text):
    if text == 'dice1':
        return 1
    elif text == 'dice2':
        return 2
    elif text == 'dice5':
        return 3
    elif text == 'dice6':
        return 4
    elif text =='torpedo':
        return 5
    else:
        return 6

def create_tf_example(filename, label_file):
    img = cv2.imread(filename)
    height, width, channels = img.shape
    
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpeg'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    with open(label_file, 'r') as f:
        csvreader = csv.reader(f, delimiter=' ')
        head = True

        for row in csvreader:
            if head:
                head = False 
                continue

            name = row[-1]
            classes_text.append(name)
            classes.append(get_index(name))

            xmins.append(float(row[0]) / width)
            xmaxs.append(float(row[2]) / width)
            ymins.append(float(row[1]) / height)
            ymaxs.append(float(row[3]) / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example



def main(img_dir, label_dir, out_file):
    imgs = glob.glob(img_dir + '/*.jpg')
    imgs.sort()
    labels = glob.glob(label_dir + '/*.txt')
    labels.sort()

    writer = tf.python_io.TFRecordWriter(out_file)

    for i, l in zip(imgs, labels):
        # print i, l
        tf_example = create_tf_example(i, l)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('img_dir', help='image directory')
    p.add_argument('label_dir', help='label directory')
    p.add_argument('out_file', help='full path of TF record file to be saved')

    args = p.parse_args()

    main(args.img_dir, args.label_dir, args.out_file)

