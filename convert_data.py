import cv2
import pandas as pd 
import numpy as np 
import os
import argparse

#dir_images = '/home/thaitran/test/test/data_train/data_taokit/augmentation_image/images'

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #y = x
    y = x.iloc[0:0]

    y['class'] = x['class']

    y['xmin'] = w * (x['xmin'] - x['xmax'] / 2) + padw  # top left x
    y['ymin'] = h * (x['ymin'] - x['ymax'] / 2) + padh  # top left y

    y['xmax'] = w * (x['xmin'] + x['xmax'] / 2) + padw  # bottom right x
    y['ymax'] = h * (x['ymin'] + x['ymax'] / 2) + padh  # bottom right y

    y = y.fillna(0)

    return y

def convert1(file_txt, class_name):

    header = ['class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.read_csv(file_txt, delimiter = " ", names= header)
    for i in range(df.shape[0]):
        if type(df['class'][i]) == str :
            continue 
        df['class'] = df['class'].replace(df['class'][i],class_name[df['class'][i]])
    df['truncation'] = df['occlusion'] = df['alpha'] = df['3dh'] = df['3dw'] = df['3dl'] = df['lox'] = df['loy'] = df['loz'] = df['roty'] = 0
    df = df[['class', 'truncation', 'occlusion', 'alpha','xmin', 
            'ymin', 'xmax', 'ymax', '3dh','3dw', '3dl', 'lox', 'loy', 'loz', 'roty']]
    return df


def convert(dir_images, dir_labels, dir_save_txt, class_name):
    list_images = os.listdir(dir_images)
    header = ['class', 'truncation', 'occlusion', 'alpha','xmin', 'ymin', 
            'xmax', 'ymax', '3dh','3dw', '3dl', 'lox', 'loy', 'loz', 'roty']

    for image in list_images:
        img = cv2.imread(os.path.join(dir_images, image))
        height, width = img.shape[:2]
        df_label = convert1(os.path.join(dir_labels, image[:-3] + 'txt'), class_name)
        data = (xywhn2xyxy(df_label, w = width, h = height))
        np.savetxt(os.path.join(dir_save_txt, (image[:-3] + 'txt')), data.values, 
            fmt='%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f')

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_img', type = str, help = 'path images')
    parser.add_argument('--dir_label', type = str, help = 'path label from LabelTool')
    parser.add_argument('--dir_txt', type = str, help = 'path save label result convert')
    parser.add_argument('--dir_class', help = 'path file class name')
    args = parser.parse_args()
    return args
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

if __name__ == "__main__":
    opt = parse()
    class_name = load_classes(opt.dir_class)
    convert(opt.dir_img, opt.dir_label, opt.dir_txt, class_name)
