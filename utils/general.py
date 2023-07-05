import glob
import logging
import math
import os
import platform
import random
import re
import sys
import urllib
from subprocess import check_output
import time
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
import pkg_resources as pkg
import torch.backends.cudnn as cudnn
from utils.google_utils import gsutil_getsize
from utils.metrics import fitness

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
# cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
# os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())  # NumExpr max threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
ONNX_OPSET = [x for x in range(11, 19)]
ONNX_OPSET_TARGET = ONNX_OPSET
CUDA = torch.cuda.is_available()
MAX_DET = 300  # topk objects for every images


def set_logging(rank=-1, filename=None, filemode=None):
    logging.basicConfig(
        filename=filename, filemode=filemode,
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def gb2mb(inp0):
    value = f"{inp0:,}"
    len_ = value.split(",")
    digit = len(len_)
    if digit <= 1:
        return f"{inp0}B"
    elif digit == 2:
        return f'{len_[0]}.{len_[1]}KB'
    elif digit == 3:
        return f"{len_[0]}.{len_[1]}MB"
    elif digit == 4:
        return f'{len_[0]}.{len_[1]}GB'
    elif digit == 5:
        return f'{len_[0]}.{len_[1]}TB'
    else:
        return f"{inp0}B"


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def init_seeds(seed=0):
    """Initialize random number generator (RNG) seeds"""
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)"""
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    """Is environment a Docker container"""
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string"""
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


def check_online():
    """Check internet connectivity"""
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accesability
        return True
    except OSError:
        return False


def check_git_status():
    """Recommend 'git pull' if code is out of date"""
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = check_output(cmd, shell=True).decode().strip().rstrip('.git')  # github repo url
        branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
        n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))  # emoji-safe
    except Exception as e:
        print(e)


def check_python(minimum='3.7.0'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLO, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        print(s)
    return result


def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=''):
    # Check installed dependencies meet YOLOv5 requirements (pass *.txt file or list of packages or single package str)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            s += f'"{r}" '
            n += 1

    if s and install:  # check environment variable
        print(f"{prefix} YOLO requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            # assert check_online(), "AutoUpdate skipped (offline)"
            print(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            print(s)
        except Exception as e:
            print(f'{prefix} ❌ {e}')


def check_img_size(img_size, s=32):
    """Verify img_size is a multiple of stride s"""
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: image size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    """Check if environment supports image displays"""
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


# def check_file(file):
#     """Search for file if not found
#         return file
#     """
#     if Path(file).is_file() or file == '':
#         return file
#     else:
#         files = glob.glob('./**/' + file, recursive=True)  # find file
#         assert len(files), f'File Not Found: {file}'  # assert file was found
#         assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
#         return files[0]  # return file


def check_dataset(dict_):
    """Download dataset if not found locally"""
    val, s = dict_.get('val'), dict_.get('download')
    if val and len(val):
        val = [Path(x) for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [x.as_posix() for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    root_dir = Path(dict_.get("path", "datasets"))
                    root_dir.mkdir(exist_ok=True)
                    file_name = Path(s).name
                    if not Path(s).exists():
                        torch.hub.download_url_to_file(s, file_name)
                    r = os.system('unzip -q %s -d %s && rm %s' % (file_name, root_dir.as_posix(), file_name))
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    elif file.startswith('clearml://'):  # ClearML Dataset ID
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def make_divisible(x, divisor):
    """Returns x evenly divisible by divisor"""
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    """Cleans a string by replacing special characters with underscore"""
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """lambda function for sinusoidal ramp from y1 to y2"""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')"""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    """Get class weights (inverse frequency) from training labels"""
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """Produces image weights based on class_weights and image contents"""
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    """https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/"""
    x = [x for x in range(1, 91)]
    return x


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalized segments into pixel segments, shape (n,2)"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    """Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)"""
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    """Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)"""
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """Up-sample an (n,2) segment"""
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[..., 0].clamp_(0, img_shape[1])  # x1
    boxes[..., 1].clamp_(0, img_shape[0])  # y1
    boxes[..., 2].clamp_(0, img_shape[1])  # x2
    boxes[..., 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Returns the IoU of box1 to box2. box1 is 4, box2 is nx4"""
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=2, eps=1e-9):
    """Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4"""
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter / union + eps, alpha)
    # beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union)  # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou  # torch.log(iou+eps) or iou


def box_iou(box1, box2):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def box_giou(box1, box2):
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def box_ciou(box1, box2, eps: float = 1e-7):
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    w_pred = box1[:, None, 2] - box1[:, None, 0]
    h_pred = box1[:, None, 3] - box1[:, None, 1]

    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]

    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (centers_distance_squared / diagonal_distance_squared) - alpha * v


def box_diou(box1, box2, eps: float = 1e-7):
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared)


def non_max_suppression(prediction: torch.Tensor, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                        multi_label=False,
                        labels=(), max_det=300, min_wh=2,
                        max_wh=4096, max_nms=30000,
                        time_limit=10.0, redundant=True, merge=False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # if prediction.device.type in ('privateuseone', "mps"):
    #     prediction = prediction.to('cpu')
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    # max_det = 300  # maximum number of detections per image
    # max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False,
                            labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5 if not kpt_label else prediction.shape[2] - 56  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5 + nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s='', halfModel=False):  # from utils.general import *; 
    """Strip optimizer from 'f' to finalize training, optionally save as 's'"""
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    if halfModel:
        x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(colorstr('Optimizer:'), f" stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.3f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    """Print mutation results to evolve.txt (for use with train.py --evolve)"""
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    """applies a second stage classifier to yolo outputs"""
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    """Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc."""
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


class BackgroundForegroundColors():
    """Custom color for bbox and text"""

    def __init__(self, names: tuple):
        """_summary_

        Args:
            names (_type_): _description_
        """
        self.COLOR = [
            [(149, 68, 1), (106, 187, 254)],
            [(246, 213, 90), (9, 42, 165)],
            [(129, 8, 165), (126, 247, 90)],
            [(105, 7, 50), (150, 248, 205)],
            [(226, 43, 36), (29, 212, 219)],
            [(191, 174, 241), (64, 81, 14)],
            [(94, 4, 245), (161, 251, 10)],
            [(20, 223, 238), (235, 32, 17)],
            [(188, 108, 255), (67, 147, 0)],
            [(251, 211, 127), (4, 44, 128)],
            [(110, 233, 181), (145, 22, 74)],
            [(29, 241, 17), (226, 14, 238)],
            [(25, 246, 126), (230, 9, 129)],
            [(82, 240, 137), (173, 15, 118)],
            [(52, 6, 109), (203, 249, 146)],
            [(254, 189, 31), (1, 66, 224)],
            [(22, 239, 132), (233, 16, 123)],
            [(179, 237, 198), (76, 18, 57)],
            [(186, 188, 84), (69, 67, 171)],
            [(43, 246, 201), (212, 9, 54)],
            [(83, 59, 137), (172, 196, 118)],
            [(160, 18, 251), (95, 237, 4)],
            [(94, 152, 205), (161, 103, 50)],
            [(71, 13, 20), (184, 242, 235)],
            [(145, 160, 225), (110, 95, 30)],
            [(202, 241, 197), (53, 14, 58)],
            [(153, 212, 70), (102, 43, 185)],
            [(215, 212, 76), (40, 43, 179)],
            [(4, 166, 3), (251, 89, 252)],
            [(5, 228, 122), (250, 27, 133)],
            [(15, 118, 8), (240, 137, 247)],
            [(182, 181, 28), (73, 74, 227)],
            [(232, 242, 249), (23, 13, 6)],
            [(91, 79, 9), (164, 176, 246)],
            [(186, 174, 223), (69, 81, 32)],
            [(152, 220, 193), (103, 35, 62)],
            [(77, 235, 251), (178, 20, 4)],
            [(224, 162, 255), (31, 93, 0)],
            [(4, 246, 179), (251, 9, 76)],
            [(4, 111, 186), (251, 144, 69)],
            [(220, 251, 181), (35, 4, 74)],
            [(168, 209, 115), (87, 46, 140)],
            [(174, 31, 65), (81, 224, 190)],
            [(21, 74, 47), (234, 181, 208)],
            [(217, 254, 107), (38, 1, 148)],
            [(109, 210, 225), (146, 45, 30)],
            [(12, 29, 0), (243, 226, 255)],
            [(241, 158, 149), (14, 97, 106)],
            [(143, 31, 219), (112, 224, 36)],
            [(36, 41, 35), (219, 214, 220)],
            [(220, 218, 88), (35, 37, 167)],
            [(49, 40, 204), (206, 215, 51)],
            [(96, 14, 228), (159, 241, 27)],
            [(221, 10, 69), (34, 245, 186)],
            [(240, 186, 152), (15, 69, 103)],
            [(177, 21, 42), (78, 234, 213)],
            [(150, 217, 142), (105, 38, 113)],
            [(43, 20, 5), (212, 235, 250)],
            [(62, 251, 143), (193, 4, 112)],
            [(178, 18, 225), (77, 237, 30)],
            [(200, 50, 40), (55, 205, 215)],
            [(226, 247, 77), (29, 8, 178)],
            [(192, 168, 160), (63, 87, 95)],
            [(2, 239, 145), (253, 16, 110)],
            [(44, 224, 86), (211, 31, 169)],
            [(14, 24, 64), (241, 231, 191)],
            [(28, 198, 187), (227, 57, 68)],
            [(73, 45, 71), (182, 210, 184)],
            [(226, 202, 96), (29, 53, 159)],
            [(5, 205, 132), (250, 50, 123)],
            [(254, 231, 152), (1, 24, 103)],
            [(139, 46, 213), (116, 209, 42)],
            [(144, 254, 140), (111, 1, 115)],
            [(160, 197, 132), (95, 58, 123)],
            [(116, 173, 0), (139, 82, 255)],
            [(49, 55, 157), (206, 200, 98)],
            [(136, 177, 251), (119, 78, 4)],
            [(217, 242, 164), (38, 13, 91)],
            [(134, 103, 2), (121, 152, 253)],
            [(248, 164, 119), (7, 91, 136)],
            [(18, 67, 66), (237, 188, 189)],
            [(117, 58, 76), (138, 197, 179)],
            [(158, 35, 66), (97, 220, 189)],
            [(18, 25, 12), (237, 230, 243)],
            [(143, 28, 166), (112, 227, 89)],
            [(143, 248, 7), (112, 7, 248)],
            [(122, 36, 95), (133, 219, 160)],
            [(61, 214, 185), (194, 41, 70)],
            [(120, 125, 29), (135, 130, 226)],
            [(130, 249, 90), (125, 6, 165)],
            [(51, 13, 94), (204, 242, 161)],
            [(223, 172, 6), (32, 83, 249)],
            [(7, 98, 127), (248, 157, 128)],
            [(118, 106, 48), (137, 149, 207)],
            [(44, 249, 127), (211, 6, 128)],
            [(158, 248, 89), (97, 7, 166)],
            [(20, 244, 125), (235, 11, 130)],
            [(245, 250, 235), (10, 5, 20)],
            [(87, 23, 215), (168, 232, 40)],
            [(66, 160, 245), (189, 95, 10)],
            [(0, 109, 72), (255, 146, 183)],
            [(26, 206, 94), (229, 49, 161)],
            [(43, 123, 13), (212, 132, 242)],
            [(180, 240, 71), (75, 15, 184)],
            [(126, 40, 9), (129, 215, 246)],
            [(173, 204, 141), (82, 51, 114)]
        ]
        self.clor = []
        for x in names:
            a = random.randint(0, len(self.COLOR) - 1)
            self.clor.append(self.COLOR[a])
            self.COLOR.remove(self.COLOR[a])

    def getval(self, index):
        """get text color, bbox color
        arg: index of class name, labels
        return: textColor, bboxColor
        """
        textColor = self.clor[index][1]
        bkColor = self.clor[index][0]
        return textColor, bkColor

    def len(self):
        """return len of COLOR[]"""
        return len(self.COLOR)


def yaml_save(file='data.yaml', data=None):
    """Single-line safe yaml saving"""
    data = {} if data is None else data
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def sliding_windows(image, stepSize, windowsSize):
    for x in range(0, image.shape[0], stepSize):
        for y in range(0, image.shape[1], stepSize):
            x_x = [x, x + windowsSize[0]]
            yield x, y, image[x:x + windowsSize[0], y:y + windowsSize[1]]
