"""
Usage:
python Making_TFrecords.py <Path_to_shelf data CSV created using Data_Prep.py> \
                    <Path_to_product data CSV created using Data_Prep.py> \

eg: python Making_TFrecords.py shelf_data.csv product_data.csv

"""

import cv2
import pandas as pd
import numpy as np
import os
import sys
import io
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# current images path 
img_path = 'GroceryDataset/ShelfImages/'
# cropped parts destination
cropped_path = 'GroceryDataset/detector/'
# Step 1 results path
data_path = '/'
# output destination
detector_data_path = 'pack_detector/data/'

N_CROP_TRIALS = 6
CROP_SIZE = 1000

# returns random value in [s, f]
def rand_between(s, f):
    if s == f:
        return s
    return np.random.randint(s, f)

def random_crop(photos,products):

    train_products, eval_products = [], []
    for img_file, is_train in photos[['image', 'train']].values:
        img = cv2.imread(f'{img_path}{img_file}')
        img_h, img_w, img_c = img.shape
        for n in range(N_CROP_TRIALS):
            # randomly crop square
            c_size = rand_between(300, max(img_h, img_w))
            x0 = rand_between(0, max(0, img_w - c_size))
            y0 = rand_between(0, max(0, img_h - c_size))
            x1 = min(img_w, x0 + c_size)
            y1 = min(img_h, y0 + c_size)
            # products totally inside crop rectangle
            crop_products = products[(products.image == img_file) & 
                                    (products.xmin > x0) & (products.xmax < x1) &
                                    (products.ymin > y0) & (products.ymax < y1)]
            # no products inside crop rectangle? cropping trial failed...
            if len(crop_products) == 0:
                continue
            # name the crop
            crop_img_file = f'{img_file[:-4]}{x0}_{y0}_{x1}_{y1}.JPG'
            # crop and reshape to CROP_SIZExCROP_SIZE or smaller 
            # keeping aspect ratio
            crop = img[y0:y1, x0:x1]
            h, w, c = crop.shape
            ratio = min(CROP_SIZE/h, CROP_SIZE/w)
            crop = cv2.resize(crop, (0,0), fx=ratio, fy=ratio)
            crop = crop[0:CROP_SIZE, 0:CROP_SIZE]
            h, w, c = crop.shape
            # add crop inner products to train_products or eval_products list
            for xmin, ymin, xmax, ymax in \
                    crop_products[['xmin', 'ymin', 'xmax', 'ymax']].values:
                xmin -= x0
                xmax -= x0
                ymin -= y0
                ymax -= y0

                xmin, xmax, ymin, ymax = [int(np.round(e * ratio)) 
                                        for e in [xmin, xmax, ymin, ymax]]
                product = {'filename': crop_img_file, 'class':'pack', 
                        'width':w, 'height':h,
                        'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}
                if is_train:
                    train_products.append(product)
                else:
                    eval_products.append(product)
            # save crop top eval or train folder
            subpath = ['eval_1/', 'train_1/'][is_train]
            cv2.imwrite(f'{cropped_path}{subpath}{crop_img_file}', crop)

    return train_products,eval_products


def class_text_to_int(row_label):
    if row_label == 'pack':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) 
            for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_to_tf_records(images_path, examples, dst_file):
    writer = tf.python_io.TFRecordWriter(dst_file)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

def main():

    # read rects and photos dataframes
    photos = pd.read_csv(sys.argv[1])
    products = pd.read_csv(sys.argv[2])

    train_products,eval_products=random_crop(photos,products)


    train_df = pd.DataFrame(train_products).set_index('filename')   
    eval_df = pd.DataFrame(eval_products).set_index('filename')

    convert_to_tf_records(f'{cropped_path}train_1/', train_df, f'{detector_data_path}train_1.record')
    convert_to_tf_records(f'{cropped_path}eval_1/', eval_df, f'{detector_data_path}eval_1.record')     

if __name__== "__main__":
    main()