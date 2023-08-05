import contextlib
import glob
import itertools
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import psutil
import torch
import torchvision
from torch.nn.functional import interpolate
from PIL import Image, ExifTags
from termcolor import colored
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.general import (check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes,
                           resample_segments, clean_str, colorstr, gb2mb)
from utils.default import IMG_FORMATS, VID_FORMATS, TQDM_BAR_FORMAT, HELP_URL, RANK, YOUTUBE
from utils.torch_utils import torch_distributed_zero_first

# Parameters

logger = logging.getLogger(__name__)
TORCH_PIN_MEMORY = False

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache='', pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, single_channel=False,
                      prefix='', shuffle=True, seed=0):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    if rect and shuffle:
        print(f'{prefix}WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache.lower(),
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      single_channel=single_channel,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))

    nd_dml = None
    try:
        import torch_directml
        nd_dml = torch_directml.device_count()
    except ImportError:
        pass
    nd = nd_dml if nd_dml else torch.cuda.device_count()
    nw = min(
        [os.cpu_count() // max(1, nd, world_size), batch_size if batch_size > 1 else 1, workers])  # number of workers
    sampler = distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed) if rank != -1 else None
    loader = DataLoader if image_weights else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)

    dataloader_ = loader(dataset,
                         batch_size=batch_size,
                         num_workers=nw,
                         sampler=sampler,
                         pin_memory=dataset.pin_memory,
                         shuffle=shuffle and sampler is None,
                         worker_init_fn=seed_worker,
                         collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                         generator=generator,
                         prefetch_factor=nw,
                         persistent_workers=True,
                         pin_memory_device='cuda' if dataset.pin_memory else '')

    return dataloader_, dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def check_data_source(source=''):
    if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')):
        result = 'stream'
    elif source.lower() in ['screen', 'this', 'monitor']:
        result = 'screen'
    else:
        result = 'media'
    return result


class LoadImages:
    """Load image from media resouce"""

    def __init__(self, path, img_size=640, stride=32, auto=True, scaleFill=False, scaleUp=True, vid_stride=1):
        """_summary_

        Args:
            path (_type_): _description_. media file's path or txt file with media path for each line
            img_size (int, optional): _description_. Defaults to 640.
            stride (int, optional): _description_. Defaults to 32. Stride of YOLO network
            auto (bool, optional): _description_. Defaults to True. set False for rectangle shape
            vid_stride (int, optional): _description_. Defaults to 1.

        Raises:
            FileNotFoundError: _description_
        """
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if Path(x).suffix in IMG_FORMATS]
        videos = [x for x in files if Path(x).suffix in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleUp = scaleUp
        self.vid_stride = vid_stride  # video frame-rate stride
        self.fps = None

        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.mode = 'image'
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        im, ratio, dwdh = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto,
                                    scaleFill=self.scaleFill)  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, {'string': s, 'frames': ([self.frames] if self.mode == 'video' else [self.nf]),
                                         'c_frame': (
                                             [self.frame] if self.mode == 'video' else [self.count])}, ratio, dwdh

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees

    def __len__(self):
        return self.nf  # number of files


class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=(640, 640), stride=32, auto=True, scaleFill=False, scaleUp=True,
                 vid_stride=1):
        """_summary_

        Args:
            sources (str, optional): _description_. Defaults to 'streams.txt'.
            img_size (int, optional): _description_. Defaults to 640.
            stride (int, optional): _description_. Defaults to 32.
            auto (bool, optional): _description_. Defaults to True.
        """

        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleUp = scaleUp
        self.vid_stride = vid_stride
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            if isinstance(sources, str):
                sources = [sources]
        n = len(sources)
        self.imgs, self.frames, self.threads = [None] * n, [0] * n, [None] * n
        self.c_frame = [0] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            if 'rtsp://' in s:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

            print(f'{i + 1}/{n}: {s}... init ')
            url = eval(s) if s.isnumeric() else s
            if urlparse(s).hostname in YOUTUBE:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                try:
                    url = pafy.new(url).getbest(preftype="mp4").url
                except Exception as ex:
                    logger.info(f"{ex}")
                    # os.system('pip install git+https://github.com/thnak/pafy.git')
            cap = cv2.VideoCapture(url, cv2.CAP_ANY)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, url]), daemon=True)
            print(f'success ({w}x{h} at {self.fps:.2f} FPS, {self.frames[i]} frames).')
            self.threads[i].start()
        print('')  # newline

    def update(self, index, cap, stream):
        # Read next stream frame in a daemon thread
        self.c_frame[index], f = 0, self.frames[index]
        while cap.isOpened() and self.c_frame[index] < f:
            self.c_frame[index] += 1
            cap.grab()
            if self.c_frame[index] % self.vid_stride == 0:  # read every 4th frame
                success, im = cap.retrieve()
                if success:
                    self.imgs[index] = im
                else:
                    self.imgs[index] = np.zeros_like(self.imgs[index])
                    cap.open(stream)
                    print('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
            if self.fps != 0:
                time.sleep(1 / self.fps)  # wait time
            else:
                time.sleep(1 / 30)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads):
            raise StopIteration
        img0 = self.imgs.copy()
        # Letterbox
        img, ratio, dwdh = [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[0]
                            for x in img0], \
            [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[1] for x in
             img0], \
            [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[2] for x in
             img0],
        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, {'frames': self.frames, 'c_frame': self.c_frame}, ratio, dwdh

    def __len__(self):
        return len(self.sources)


class LoadScreenshots:
    def __init__(self, source, img_size=(640, 640), stride=32, auto=True, scaleFill=False, scaleUp=True):
        try:
            check_requirements('mss')
            import mss
        except ImportError as ie:
            print(f'{ie}')
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleUp = scaleUp
        self.mode = 'screen'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
        self.fps = 30

    def __iter__(self):
        return self

    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        img0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "
        img, ratio, dwdh = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto, scaleFill=self.scaleFill,
                                     scaleUp=self.scaleUp)  # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        self.frame += 1
        return str(self.screen), img, img0, None, {'string': s, 'frames': [1], 'c_frame': [1]}, ratio, dwdh


def img2label_paths(img_paths):
    """Define label paths as a function of image paths"""
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


# start for images/videos classify model

def create_dataloader_cls(dataset, batch_size,
                          rank=-1, world_size=1, workers=8,
                          prefix='', shuffle=True, seed=0):
    batch_size = min(batch_size, len(dataset))
    nd_dml = None

    try:
        import torch_directml
        nd_dml = torch_directml.device_count()
    except ImportError:
        pass
    nd = nd_dml if nd_dml else torch.cuda.device_count()
    nw = min(
        [os.cpu_count() // max(1, nd, world_size), batch_size if batch_size > 1 else 1, workers])  # number of workers
    sampler = distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed) if rank != -1 else None
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    if dataset.pin_memory:
        print(f'{prefix} PyTorch {torch.__version__} with pin_memory is enable, it will use your RAM')
    dataset.prepare()
    the_dataloader = InfiniteDataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle and sampler is None,
                                        num_workers=nw,
                                        sampler=sampler,
                                        pin_memory=dataset.pin_memory,
                                        worker_init_fn=seed_worker,
                                        generator=generator)

    return the_dataloader


class LoadSampleAndTarget(torchvision.datasets.ImageFolder):
    version = 0.1
    image_8bit = True
    minimum_size = 100
    std = [1., 1., 1.]
    mean = [0., 0., 0.]

    def __init__(self, root, hyp=None, augment=True, cache=True, prefix="", backend='pyav'):
        super().__init__(root=root)
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.prefix = prefix

        self.pin_memory = False
        self.imgsz = 224  # it will be changes when init model
        self.augment = augment,
        self.size = 224
        self.scale = (0.08, 1.0)
        self.ratio = (0.75, 1.0 / 0.75)  # 0.75, 1.33
        self.hflip = 0.25
        self.vflip = 0.25
        self.jitter = 0.4
        total_caching = self.find_value_canbe_cache(prefix=prefix)
        self.total_caching = total_caching
        self.auto_aug = False
        self.transform = None
        self.cache = cache

    def calculatingMeanSTD(self, max_number, prefix):
        """calculate mean, std"""
        # task = []
        imgsz = (self.imgsz // 2, self.imgsz // 2)
        pbar = tqdm(range(max_number), total=max_number)
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        for x in enumerate(pbar):
            img = self.loadImage(x)[0]
            img = cv2.resize(img, imgsz, interpolation=cv2.INTER_NEAREST)
            img = img[:, :, ::-1].astype(np.float32)  # BGR to RGB
            img /= 255.
            img = torch.from_numpy(img)
            img = torch.unsqueeze(img, dim=0)
            img = torch.permute(img, [0, 3, 1, 2])  # -> BCHW
            img = img.float()
            psum += img.sum(axis=[0, 2, 3])
            psum_sq += (img ** 2).sum(axis=[0, 2, 3])

            # task.append(img)
            pbar.set_description(f"{prefix}Collecting data to calculate mean, std...")
            if x >= max_number - 10:
                count = (x + 1) * self.imgsz * self.imgsz
                # mean and std
                total_mean = psum / count
                total_var = (psum_sq / count) - (total_mean ** 2)
                total_std = torch.sqrt(total_var)
                self.mean = total_mean.cpu().numpy().tolist()
                self.std = total_std.cpu().numpy().tolist()
                pbar.set_description(f"{prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")

    def dataset_analysis(self):
        dick = {}
        for x in range(len(self.classes)):
            dick[x] = 0
        for x in range(len(self.samples)):
            img, label = self.loadImage(x)
            dick[label] += 1
        return dick, self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, j = self.loadImage(index)
        img = img[:, :, ::-1]  # BGR to RGB
        img = self.transform(image=img)["image"]
        return img, j

    def loadImage(self, index):
        """Load image and label"""
        f, j, fn, im = self.samples[index]  # filename, index, filename.with_suffix('.npy'), image
        if im is None:
            if fn.exists():
                img = np.load(fn.as_posix())
            else:
                img = cv2.imread(f)  # BGR
                h, w, c = img.shape
                r = max(h, w)
                if r > self.imgsz:
                    ratio = self.imgsz / r
                    img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        else:
            img = im.copy()
        return img, j

    def find_value_canbe_cache(self, prefix="", safety_margin=1.5):
        total = len(self.samples)
        nbytes = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8).nbytes
        total_nbytes = nbytes * safety_margin
        mem = psutil.virtual_memory()
        free = mem.available
        mem_2_caching = int(free / total_nbytes)
        mem_2_caching = min(mem_2_caching, total)
        logger.info(f"{prefix}Total {mem_2_caching} in {total} images can "
                    f"be cache in memory with total {gb2mb(total_nbytes * mem_2_caching)}")
        return mem_2_caching

    def prepare(self):
        if sum(self.mean) == 0 and sum(self.std) == 3:
            self.calculatingMeanSTD(len(self.samples), self.prefix)
        if self.cache:
            pbar = tqdm(range(0, self.total_caching), desc="%33s" % "", total=self.total_caching)
            gb = 0
            for i, x in enumerate(pbar):
                self.samples[x][3] = self.loadImage(x)[0]
                gb += self.samples[x][3].nbytes
                pbar.set_description(f"{self.prefix}Caching... {gb2mb(gb)}")
                if i == self.total_caching - 1:
                    pbar.set_description(f"{self.prefix}Cached {gb2mb(gb)}")
        self.transform = self.classify_albumentations()

    def classify_albumentations(self):
        prefix = self.prefix
        augment = self.augment
        size = self.imgsz
        scale = self.scale
        ratio = self.ratio
        auto_aug = self.auto_aug
        hflip = self.hflip
        vflip = self.vflip
        jitter = self.jitter
        std, mean = self.std, self.mean
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                logger.info(f'{prefix}auto augmentations are currently not supported')
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std)]
        T += [ToTensorV2()]  # Normalize and convert to Tensor
        logger.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        return A.Compose(T)

    @staticmethod
    def collect_fn(batch):
        images, target = zip(*batch)
        return torch.stack(images, 0), torch.tensor(target, dtype=torch.long)


# end for image classify model

# start for video classify model

class Load_Sample_for_Video_Classify(Dataset):
    VID_FORMATS = ['.asf', '.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv',
                   '.gif']  # acceptable video suffixes
    mean = (0., 0., 0.)
    std = (1., 1., 1.)
    use_BGR = False

    def __init__(self, root, hyp=None, augment=True, cache=True, prefix="", backend='pyav'):
        self.transform_2 = None
        self.transform = None
        self.root = root = Path(root) if isinstance(root, str) else root
        if augment:
            assert isinstance(hyp, dict), f"{prefix}use augment but hyper parameter not found."
        self.augment = hyp
        self.prefix = prefix
        from torchvision.datasets.folder import make_dataset
        classes = [x.name for x in root.iterdir() if x.is_dir()]
        classes.sort()
        class_to_indx = {class_name: i for i, class_name in enumerate(classes)}
        self.classes = classes
        self.class_to_indx = class_to_indx
        assert len(class_to_indx) > 0, f"dataset not found ({root.as_posix()})"
        self.samples = make_dataset(directory=root.as_posix(),
                                    class_to_idx=class_to_indx,
                                    extensions=tuple(VID_FORMATS))
        self.imgsz = 224
        self.sample_length = 16
        self.sampling_rate = 5
        self.pin_memory = False
        backend = backend.lower()
        try:
            if backend == "pyav":
                check_requirements("av")
            torchvision.set_video_backend(backend=backend)
        except Exception as ex:
            check_requirements("av")
            torchvision.set_video_backend("pyav")
            logger.info(f"{self.prefix}fallback to PyAV backend \n{ex}")

    def prepare(self):
        """prepare dataset"""
        if sum(self.mean) == 0 and sum(self.std) == 3:
            self.calculateMeanStd()
        else:
            logger.info(f"{self.prefix}Using mean: {self.mean}, std: {self.std} from model.yaml for this dataset.")
        self.sampling_rate = max(1, int(self.sampling_rate))
        self.sample_length = max(1, int(self.sample_length))

        compose = []
        if self.augment:
            augment = self.augment
            ranDropChannel = augment.get('ChannelDropout', 0)
            ranDropFrame = augment.get("FrameDrop", 0)
            h_flip, v_flip = augment.get("HorizontalFlip", 0), augment.get("VerticalFlip", 0)
            bright, contrast = augment.get("brightness_limit"), augment.get("contrast_limit", 0)
            satu = augment.get("saturation_limit", 0)
            gray = augment.get("toGray", 0)
            rotate = augment.get("degrees", 0)
            randomRotate = augment.get("rotate", 0)
            channelShuffle = augment.get("channelShuffle", 0)

            compose.extend([transforms.Lambda(lambd=lambda x: self.randomDropChannel(x, p=ranDropChannel)),
                            transforms.Lambda(lambd=lambda x: self.randomDropFrame(x, p=ranDropFrame)),
                            transforms.Lambda(lambd=lambda x: self.channelShuffle(x, p=channelShuffle)),
                            transforms.Lambda(lambd=lambda x: self.h_flip(x, p=h_flip)),
                            transforms.Lambda(lambd=lambda x: self.v_flip(x, p=v_flip)),
                            transforms.Lambda(lambd=lambda x: self.RandomRotate(x, rotate, p=randomRotate)),
                            transforms.Lambda(lambd=lambda x: self.ColorJitter(x, bright,
                                                                               contrast, satu,
                                                                               hue=0, p=0.1)),
                            transforms.Lambda(lambd=lambda x: self.rgb_2_gray(x, p=gray))])
            logger.info(f"{self.prefix}Using {augment}")
            self.transform_2 = transforms.Compose(compose)
        compose2 = compose + [transforms.Lambda(lambd=lambda x: self.normalizeInputs(x, self.mean, self.std))]
        self.transform = transforms.Compose(compose2)

        logger.info(
            f"{self.prefix}total {len(self.samples)} samples with {len(self.classes)} classes, "
            f"frame length: {self.sample_length}, step frame: {self.sampling_rate}")

    def dataset_analysis(self):
        """for now only return number of frame per classes"""
        dict_ = {target: 0 for sample, target in self.samples}
        for x, i in self.samples:
            vid = torchvision.io.VideoReader(x, "video")
            vid.set_current_stream("video")

            metadata = vid.get_metadata()
            total_frames = metadata["video"]['duration'][0] * metadata["video"]['fps'][0]
            dict_[i] += total_frames
        return dict_, self.classes

    def calculateMeanStd(self):
        """Calculate mean, std. https://kozodoi.me/blog/20210308/compute-image-stats"""
        samples = self.samples
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        pbar = tqdm(samples, total=len(samples))
        transform = transforms.Compose([
            transforms.Lambda(lambd=lambda x: self.normalizeInputs(x, mean=[0.]*3, std=[1.]*3))])

        for i, (path, _) in enumerate(pbar):
            video, _ = self.loadSample(path, transform=transform, dtype=torch.float32)
            psum += video.sum(axis=[0, 2, 3])
            psum_sq += (video ** 2).sum(axis=[0, 2, 3])
            pbar.set_description(f"{self.prefix}Collecting data to calculate mean, std...")
            if i >= len(samples) - 10:
                count = (i + 1) * self.sample_length * self.imgsz * self.imgsz
                total_mean = psum / count
                total_var = (psum_sq / count) - (total_mean ** 2)
                total_std = torch.sqrt(total_var)
                self.mean = total_mean.cpu().numpy().tolist()
                self.std = total_std.cpu().numpy().tolist()
                pbar.set_description(f"{self.prefix}Calculating...")
                pbar.set_description(f"{self.prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        video = self.loadSample(path=path, transform=self.transform, dtype=torch.float32)[0]
        video = torch.permute(video, dims=[1, 0, 2, 3])  # NCHW -> CNHW
        return video, target

    def loadSample(self, path=None, transform=None, dtype=torch.uint8) -> tuple[torch.Tensor, str]:
        """load sample and apply transform"""
        target = 0
        resize_transform = transforms.Compose([transforms.Resize((self.imgsz, self.imgsz), antialias=False)])
        if path is None:
            path, target = random.choice(self.samples)
        vid = torchvision.io.VideoReader(path, "video")
        vid.set_current_stream("video")
        metadata = vid.get_metadata()
        # Seek and return frames
        n_length = self.sample_length * self.sampling_rate
        fps = metadata["video"]['fps']
        fps = fps[0] if isinstance(fps, list) else fps
        max_seek = metadata["video"]['duration'][0] - (n_length / fps)
        start = random.uniform(0., max_seek)
        video = torch.zeros([self.sample_length, 3, self.imgsz, self.imgsz], dtype=dtype)
        for i, frame in enumerate(itertools.islice(vid.seek(start, keyframes_only=True), 0,
                                                   n_length, self.sampling_rate)):
            video[i, ...] = resize_transform(frame['data'])

        if transform:
            video = transform(video)
        return video, self.classes[target]

    @staticmethod
    def rgb_2_gray(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        """convert multiple rgb image to gray"""
        gray = inputs.float()
        n_shape = inputs.dim()
        if random.random() <= p or always_apply:
            if n_shape == 4:
                for i, rgb in enumerate(gray):
                    r, g, b = rgb[0, ...], rgb[1, ...], rgb[2, ...]
                    g = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    gray[i, ...] = torch.stack([g, g, g])
            elif n_shape == 3:
                r, g, b = gray[0, ...], gray[1, ...], gray[2, ...]
                g = 0.2989 * r + 0.5870 * g + 0.1140 * b
                gray = torch.stack([g, g, g])
            else:
                raise f"{n_shape} dimension does not support."
        return gray.round().to(torch.uint8)

    @staticmethod
    def randomDropFrame(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        if random.random() <= p or always_apply:
            n_dims = inputs.dim()
            fill_value = random.uniform(0, 255)
            if n_dims == 4:
                frame_idx = random.randint(0, inputs.shape[0] - 1)
                inputs[frame_idx, ...] = int(fill_value)
            else:
                inputs[...] = int(fill_value)
        return inputs

    @staticmethod
    def randomDropChannel(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        if random.random() <= p or always_apply:
            channel = random.randint(0, 2)
            n_dims = inputs.dim()
            fill_value = random.uniform(0, 10)
            if n_dims == 4:
                inputs[:, channel, ...] = int(fill_value)
            else:
                inputs[channel, ...] = int(fill_value)

        return inputs

    @staticmethod
    def v_flip(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        if random.random() <= p or always_apply:
            inputs = F.hflip(inputs)
        return inputs

    @staticmethod
    def h_flip(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        if random.random() <= p or always_apply:
            inputs = F.vflip(inputs)
        return inputs

    @staticmethod
    def channelShuffle(inputs: torch.Tensor, always_apply=False, p=0.5) -> torch.Tensor:
        if random.random() <= p or always_apply:
            n_dims = inputs.dim()
            if n_dims >= 4:
                r, g, b = inputs[..., 0, :, :], inputs[..., 1, :, :], inputs[..., 2, :, :]
                r, g, b = r.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0)
                shuffled = [r, g, b]
                random.shuffle(shuffled)
                inputs = torch.concat(shuffled, 0)
                inputs = inputs.permute([1, 0, 2, 3])
            else:
                r, g, b = inputs[0, ...], inputs[1, ...], inputs[2, ...]
                r, g, b = r.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0)
                shuffled = [r, g, b]
                random.shuffle(shuffled)
                inputs = torch.concat(shuffled, 0)
                inputs = inputs.permute([1, 0, 2])

        return inputs

    @staticmethod
    def normalizeInputs(inputs: torch.Tensor,
                        mean: tuple[float, float, float] | tuple[float],
                        std: tuple[float, float, float] | tuple[float],
                        pixel_max_value=255.) -> torch.FloatTensor:
        """input int tensor in NCHW or CHW format and return the same format"""
        n_dims = inputs.dim()
        inputs = inputs.float()
        inputs /= pixel_max_value
        if n_dims == 4:
            n_dept, c, h, w = inputs.shape
            assert c == len(mean), f"len of mean ({len(mean)}) must be equal to image channel ({c})"
            for n in range(n_dept):
                for x in range(c):
                    inputs[n, x, ...] = (inputs[n, x, ...] - mean[x]) / std[x]
        elif n_dims == 3:
            c, h, w = inputs.shape
            assert c == len(mean), f"len of mean ({len(mean)}) must be equal to image channel ({c})"
            for x in range(c):
                inputs[x, ...] = (inputs[x, ...] - mean[x]) / std[x]
        else:
            raise f"inputs tensor must be 3 or 4 dimension, got {n_dims}"
        return inputs

    @staticmethod
    def ColorJitter(inputs: torch.Tensor,
                    brightness=0.2,
                    contrast=0.2, saturation=0.2, hue=0.2,
                    always_apply=False, p=0.5):
        """
        value = brightness = contrast = saturation = 1 is origin image
        0 < value < 1 is lower
        value > 1 is higher
        for hue must in range [-0.5, 0.5]

        always_apply=True use this augment for all the time
        """
        brightness = max(brightness, 0)
        contrast = max(contrast, 0)
        saturation = max(saturation, 0)
        hue = min(hue, 0.5)
        hue = max(hue, 0)

        assert 0 <= p <= 1, f'Probability must in range [0, 1]'

        if random.random() <= p or always_apply:
            brightness = random.uniform(1 - brightness, 1 + brightness)
            if brightness != 1:
                inputs = F.adjust_brightness(inputs, brightness)

            contrast = random.uniform(1 - contrast, 1 + contrast)
            if contrast != 1:
                inputs = F.adjust_contrast(inputs, contrast)

            saturation = random.uniform(1 - saturation, 1 + saturation)
            if saturation != 1:
                inputs = F.adjust_saturation(inputs, saturation)

            hue = random.uniform(-hue, hue)
            if hue != 0:
                inputs = F.adjust_hue(inputs, hue)
        return inputs

    @staticmethod
    def RandomRotate(inputs: torch.Tensor, angle: float, always_apply=False, p=0.5):
        if random.random() <= p or always_apply:
            angle = random.uniform(0, angle)
            fill_val = int(random.uniform(0, 255))
            inputs = F.rotate(inputs, angle, fill=fill_val)
        return inputs


# end for video classify model


class LoadImagesAndLabels(Dataset):  # for training/testing
    version = 0.1
    image_8bit = True
    minimum_size = 100

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images='ram', single_cls=False, stride=32, pad=0.0, single_channel=False, prefix=''):

        self.is_labeled = []
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(hyp, rect=rect) if augment else None
        self.pin_memory = False
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if Path(x).suffix in IMG_FORMATS])
            assert self.img_files, f'{prefix}No images found from {path}'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or cache[
                'version'] != self.version:  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        self.is_labeled = cache.get("labeled", [])
        cache.pop("labeled")
        nf, nb, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nb} background, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n, mininterval=0.05, maxinterval=1,
                 unit='image', bar_format=TQDM_BAR_FORMAT)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        self.imgs = [None] * (nf + nb)
        self.img_npy = [None] * (nf + nb)
        self.ignores = nb + ne + nc
        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float32)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i_rect = ar.argsort()
            self.img_files = [self.img_files[i] for i in i_rect]
            self.label_files = [self.label_files[i] for i in i_rect]
            self.labels = [self.labels[i] for i in i_rect]
            self.shapes = s[i_rect]  # wh
            ar = ar[i_rect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

        self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
        self.im_cache_dir.mkdir(parents=True, exist_ok=True)
        self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images in ['ram', 'disk']:
            ram_idx, disk_idx = self.check_cache_ram(prefix=prefix)
            if torch.cuda.is_available():
                self.pin_memory = TORCH_PIN_MEMORY

            gb = 0  # Gigabytes of cached images

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool().imap(self.load_image, range(n))
            pbar = tqdm(results, total=n, unit='image', bar_format=TQDM_BAR_FORMAT)
            checkImgSizeStatus = False
            for i, x in enumerate(pbar):
                if not self.is_labeled[i]:
                    continue
                if i < ram_idx:
                    cache_images = "ram"
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                elif i < disk_idx:
                    cache_images = "disk"
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    else:
                        if not checkImgSizeStatus:
                            testSize = np.load(self.img_npy[i].as_posix())
                            assert self.img_size in testSize.shape[:2], colored(
                                f'You need to re-cache dataset by remove folder {self.im_cache_dir}', 'red')
                            checkImgSizeStatus = True
                    gb += os.path.getsize(self.img_npy[i])
                else:
                    cache_images = ""
                    continue
                pbar.set_description(f'{prefix}Caching images {gb2mb(gb)} in {cache_images.upper()}')
            pbar.close()

    def check_cache_ram(self, prefix='', safety_margin=1.5):
        tem = int(self.stride * ((self.img_size / self.stride) - 1))
        img_sz = [self.img_size] * 2 if not self.rect else [self.img_size, tem]
        num = np.zeros((*img_sz, 3), dtype=np.uint8 if self.image_8bit else np.uint16)
        files_canbe_cache = len(self.img_files) - self.ignores
        b = num.nbytes * files_canbe_cache
        mem = psutil.virtual_memory()
        num2cache_ram = mem.available * (mem.available / (mem.available * safety_margin))
        num2cache_ram = min(int(num2cache_ram / num.nbytes), files_canbe_cache)
        import shutil
        total_disk, used_disk, free_disk = shutil.disk_usage(Path(__file__).as_posix())
        num2cache_disk = free_disk * (free_disk / (free_disk * safety_margin))
        num2cache_disk = min(int(num2cache_disk / num.nbytes), files_canbe_cache - num2cache_ram)

        print(f"{prefix}{gb2mb(b)} memory required (estimate), "
              f"ram {gb2mb(mem.available)}/{gb2mb(mem.total)} available, disk {gb2mb(free_disk)}/{gb2mb(total_disk)} "
              f"available.\n"
              f"{prefix}Caching {num2cache_ram} in ram, {num2cache_disk} in disk.")
        return num2cache_ram, num2cache_disk

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """Cache dataset labels, check images and read shapes"""
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files),
                    desc='Scanning images',
                    total=len(self.img_files),
                    mininterval=0.05,
                    maxinterval=1,
                    unit='obj',
                    bar_format=TQDM_BAR_FORMAT)
        self.is_labeled = [False] * len(self.img_files)
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert shape[0] * shape[1] > self.minimum_size, f'image size {shape} < {self.minimum_size} pixels'
                assert Path(
                    im_file).suffix in IMG_FORMATS, f'invalid image format {im.format} the format must be {IMG_FORMATS}'
                del im
                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    self.is_labeled[i] = True
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        self.is_labeled[i] = False
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    self.is_labeled[i] = False
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                self.is_labeled[i] = False
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} background, {ne} empty, {nc} corrupted"
        pbar.close()

        assert nf != 0, f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}'
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = self.version  # cache version
        x["labeled"] = self.is_labeled
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def load_image(self, index):
        """Load image from disk if not in cached or from cached if use --cache RAM"""
        img = self.imgs[index]
        img_npy = self.img_npy[index]
        if img is None:
            if img_npy.exists():
                img = np.load(img_npy.as_posix())
                return img, img.shape[:2], img.shape[:2]

            else:
                path = self.img_files[index]
                img = cv2.imread(path, cv2.IMREAD_COLOR)  # ignore alpha channel
                assert img is not None, 'Image Not Found ' + path
                h0, w0 = img.shape[:2]
                r = self.img_size / max(h0, w0)
                if r != 1:
                    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
                return img, (h0, w0), img.shape[:2]
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]

    def __len__(self):
        """Return number of valid image include background"""
        return len(self.img_files)

    def __getitem__(self, index):
        """Torch get item for training"""
        index = self.indices[index]  # linear, shuffled, or image_weights

        mosaic = True if self.mosaic and random.random() < self.hyp['mosaic'] else False
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = self.load_mosaic(index)
            else:
                img, labels = self.load_mosaic9(index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < self.hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = self.load_mosaic(random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = self.load_mosaic9(random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleUp=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        nL = len(labels)
        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=self.hyp['degrees'],
                                                 translate=self.hyp['translate'],
                                                 scale=self.hyp['scale'],
                                                 shear=self.hyp['shear'],
                                                 perspective=self.hyp['perspective'])
            img, labels = self.albumentations(img, labels)
            nL = len(labels)

            if random.random() < self.hyp['cutout']:
                img, labels = cutout(img, labels)

            if random.random() < self.hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:
                    sample_labels_, sample_images_, sample_masks_ = self.load_samples(random.randint(0,
                                                                                                     len(self.labels) - 1))
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    if len(sample_labels) == 0:
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return deepcopy(torch.from_numpy(img)), deepcopy(labels_out), deepcopy(self.img_files[index]), deepcopy(shapes)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

    def load_mosaic(self, index):
        """loads images in a 4-mosaic"""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4, labels4, segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        """loads images in a 9-mosaic"""

        img9 = None
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        # img9, labels9, segments9 = remove_background(img9, labels9, segments9)
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste'])
        img9, labels9 = random_perspective(img9, labels9, segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    def load_samples(self, index):
        """loads images in a 4-mosaic"""

        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = remove_background(img4, labels4, segments4)
        sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

        return sample_labels, sample_images, sample_masks


def copy_paste(img, labels, segments, probability=0.5):
    """SEE Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)"""
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)

        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0, w - 1), l[2].astype(int).clip(0, h - 1), l[3].astype(int).clip(0, w - 1), l[
                4].astype(int).clip(0, h - 1)

            # print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue

            sample_labels.append(l[0])

            mask = np.zeros(img.shape, np.uint8)

            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])

            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            # print(box)
            sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleUp=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]
    image2 = image
    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image2[ymin:ymax, xmin:xmax] = [random.randint(0, 255) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.65]  # remove >60% obscured labels

    return image2, labels


def pastein(image, labels, sample_labels, sample_images, sample_masks):
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        else:
            ioa = np.zeros(1)

        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (
                ymax > ymin + 20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels) - 1)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
            r_w = int(ws * r_scale)
            r_h = int(hs * r_scale)

            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])

                    image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop

    return labels


class Albumentations:
    """Data augmentation with Albumentations library"""

    def __init__(self, hyp=None, rect=True):
        check_requirements('albumentations')
        import albumentations as A
        self.transform = A.Compose([
            A.CLAHE(p=hyp['CLAHE'],
                    clip_limit=hyp['CLAHE_clip_limit'],
                    tile_grid_size=(32, 32)),
            A.RandomBrightnessContrast(brightness_limit=hyp['brightness_limit'],
                                       contrast_limit=hyp['contrast_limit'],
                                       p=hyp['RandomBrightnessContrast'],
                                       brightness_by_max=hyp['brightness_by_max']),
            A.MedianBlur(p=hyp['MedianBlur'],
                         blur_limit=hyp['MedianBlur_blur_limit']),
            A.ToGray(p=hyp['toGray']),
            A.ImageCompression(quality_lower=hyp['ImageCompression_quality_lower'],
                               quality_upper=100, p=hyp['ImageCompression']),
            A.ISONoise(p=hyp['ISONoise']),
            A.RandomRotate90(p=hyp['RandomRotate90'] if rect is False else 0),
            A.HueSaturationValue(hue_shift_limit=hyp['HueSaturationValue_hue_shift_limit'],
                                 sat_shift_limit=hyp['HueSaturationValue_sat_shift_limit'],
                                 val_shift_limit=hyp['HueSaturationValue_val_shift_limit'],
                                 p=hyp['HueSaturationValue']),
            A.HorizontalFlip(p=hyp['HorizontalFlip']),
            A.VerticalFlip(p=hyp['VerticalFlip']),
            A.ChannelDropout(p=hyp['ChannelDropout']),
            A.PixelDropout(p=hyp['PixelDropout'], drop_value=None)],

            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, labels):
        """
        Args:
            im (Image): opencv image
            labels (labels): passcal_voc

        Returns:
            image, labels
        """
        if self.transform:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def create_folder(path='./new'):
    """Create folder"""
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco'):
    """Flatten a recursive directory by bringing all files to top level"""
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco/'):
    """Convert detection dataset into classification dataset, with one directory per class"""

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int32)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if Path(x).suffix in IMG_FORMATS)  # image files only
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


def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    return self.segs[key]


def seed_worker(worker_id):
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_hash(files):
    """Returns a single hash value of a list of files"""
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    """Returns exif-corrected PIL size"""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s
