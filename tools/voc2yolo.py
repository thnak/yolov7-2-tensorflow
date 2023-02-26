from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from yaml.loader import SafeLoader
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import os
import sys
from pathlib import Path
import random
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

"""
Organize your directory of custom dataset as follows:
pull https://github.com/meituan/YOLOv6/pull/671
    data_custom
    ├───train
    │   ├───images
    │   │   ├───image_xxx.png
    │   │   └───image_xxx.png ...
    │   ├───labels_xml
    │   │   ├───label_xxx.xml
    │   │   └───label_xxx.xml ...
    │   └───labels_txt
    ├───test
    │   ├───images
    │   │   ├───image_xxx.png
    │   │   └───image_xxx.png ...
    │   ├───labels_xml
    │   │   ├───label_xxx.xml
    │   │   └───label_xxx.xml ...
    │   └───labels_txt
    └───val
        ├───images
        │   ├───image_xxx.png
        │   └───image_xxx.png ...
        ├───labels_xml
        │   ├───label_xxx.xml
        │   └───label_xxx.xml ...
        └───labels_txt
"""


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(xml_lb_dir, txt_lb_dir, image_id, voc_names):
    in_file = open(os.path.join(xml_lb_dir, f'{image_id}.xml'))
    out_file = open(os.path.join(txt_lb_dir, f'{image_id}.txt'), 'w')
    tree = ElementTree.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in voc_names and not int(obj.find('difficult').text) == 1:
            xml_box = obj.find('bndbox')
            bb = convert_box((w, h), [float(xml_box.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = voc_names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


def convert_labels_xml2txt(path_data, voc_class):
    dir_labels_txt = os.path.join(path_data, 'labels_txt')
    dir_labels_xml = os.path.join(path_data, 'labels_xml')
    if not os.path.exists(dir_labels_txt):
        os.mkdir(dir_labels_txt)
        print("created folder : ", dir_labels_txt)
    list_labels_xml = os.listdir(dir_labels_xml)
    for i in tqdm(range(0, len(list_labels_xml))):
        file_id = list_labels_xml[i].strip().split(".")[0]
        convert_label(xml_lb_dir=dir_labels_xml, txt_lb_dir=dir_labels_txt, image_id=file_id, voc_names=voc_class)
        pass
    pass


def runs(dict_folder, voc_name_of_class):
    for folder in dict_folder:
        if os.path.exists(dict_folder[folder]):
            print("========== Start processing folder", dict_folder[folder], "============")
            convert_labels_xml2txt(path_data=dict_folder[folder], voc_class=voc_name_of_class)
            print("========== End processing folder", dict_folder[folder], "============")
        else:
            print("No such file or directory: ", dict_folder[folder])
    pass


def main(config):

    data_yaml = config
    if isinstance(data_yaml, str):
        with open(data_yaml, errors='ignore') as f:
            data_dict = yaml.load(f, Loader=SafeLoader)
            runs(
                dict_folder={
                    'train': data_dict['train'],
                    'val': data_dict['val'],
                    'test': data_dict['test'],
                },
                voc_name_of_class=data_dict['names']
            )


def autosplit(path='coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'pfm']  # acceptable image suffixes

    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_yaml', type=str, default='', help='from root to path data.yaml')
    parser.add_argument('--split-path', type=str, help='path')
    parser.add_argument('--split-rate', type=float, nargs='+', help='split rate [train val test]')
    args = parser.parse_args()
    if args.path_yaml:
        print("==========start convert labels VOC xml to txt=============")
        main(args.path_yaml)
        print("==========End convert labels VOC xml to txt=============")
    if args.split_path and args.split_rate:
        autosplit(args.split_path, args.split_rate)
