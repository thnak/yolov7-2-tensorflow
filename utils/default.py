import platform
from pathlib import Path

import torch
from torch import nn as nn

from utils.activations import Hardswish, SiLU
import numpy as np
import pandas as pd
import os

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACT_LIST = (nn.LeakyReLU, nn.Hardswish, Hardswish, nn.ReLU, nn.ReLU6,
            nn.SiLU, SiLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.PReLU,
            nn.Softmax, nn.Hardsigmoid, nn.GELU, nn.Softsign, nn.Softplus)
IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo', '.pfm', '.jpg', '.jpeg',
               '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']  # acceptable image suffixes
VID_FORMATS = ['.asf', '.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv',
               '.gif']
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
YOUTUBE = ('www.youtube.com', 'youtube.com', 'youtu.be', 'https://youtu.be')

UP_SAMPLE_MODES = ['nearest', 'linear', 'bilinear', 'bicubic']
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv7 root directory
ONNX_OPSET = [x for x in range(11, 19)]
ONNX_OPSET_TARGET = 12
CUDA = torch.cuda.is_available()
MAX_DET = 300  # top-k objects for every images
RANK = int(os.getenv('RANK', -1))

MACOS = platform.system() == 'Darwin'  # macOS environment
