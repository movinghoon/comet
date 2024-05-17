import os
import random
import numpy as np
import PIL
import pickle
from PIL import Image
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import h5py
import json


# from .taskonomy_constants import SEMSEG_CLASSES, SEMSEG_CLASS_RANGE, TASKS_GROUP_DICT, TASKS, BUILDINGS
# from .augmentation import RandomHorizontalFlip, FILTERING_AUGMENTATIONS, RandomCompose, Mixup
# from .utils import crop_arrays, SobelEdgeDetector

def _load_data_hdf5(
        data_path: Path, metadata_suffix: str = "metadata.npy"):
    """Loads data and metadata assuming the data is hdf5, and converts it to dict."""
    # metadata_fname = f"{data_path.stem.split('-')[0]}-{metadata_suffix}"
    # metadata_path = data_path.parent / metadata_fname
    # metadata = np.load(str(metadata_path), allow_pickle=True).item()
    # if not isinstance(metadata, dict):
    #     raise RuntimeError(f"Metadata type {type(metadata)}, expected instance of dict")
    dataset = h5py.File(data_path, "r")
    # From `h5py.File` to a dict of `h5py.Datasets`.
    dataset = {k: dataset[k] for k in dataset}
    # return dataset, metadata
    return dataset, None

class FFHQ_256(Dataset):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.load_npy=load_npy

        self.img_paths = sorted([x for x in Path(root_dir).glob('*.png')])
        print(f'length of the dataset : {len(self.img_paths)}')

        if dset_size is None:
            self.dset_size = len(self.img_paths)
        else:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def transform_vit(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize(self.vit_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, len(self.img_paths)-1)

        if self.load_npy:
            mean = self.latent_mean[_idx]
            std = self.latent_std[_idx]
            mean = torch.as_tensor(mean, dtype=torch.float32)
            std = torch.as_tensor(std, dtype=torch.float32)
            img  = Image.open(self.img_paths[_idx]).convert('RGB')
            img = self.transform(img)
            return {'mean':mean, 'std':std, 'image':img}
        else:
            img  = Image.open(self.img_paths[_idx]).convert('RGB')

        return {'image':self.transform(img)}

class CelebA_256(FFHQ_256):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.load_npy=load_npy

        self.img_paths = sorted([x for x in Path(root_dir).glob('*.jpg')])
        print(f'length of the dataset : {len(self.img_paths)}')

        if dset_size is None:
            self.dset_size = len(self.img_paths)
        else:
            self.dset_size = dset_size



# DATAset for BDD100k dataset
class BDD100k(Dataset):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.load_npy=load_npy

        self.img_paths = sorted([x for x in Path(os.path.join(root_dir,f'images/100k/{split}')).glob('*.jpg')])
        print(f'length of the dataset : {len(self.img_paths)}')

        if dset_size is None:
            self.dset_size = len(self.img_paths)
        else:
            self.dset_size = dset_size


    def __len__(self):
        return self.dset_size
        # return len(self.img_paths)

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(720),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def transform_vit(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(720),
            transforms.Resize(self.vit_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, len(self.img_paths)-1)

        if self.load_npy:
            mean = self.latent_mean[_idx]
            std = self.latent_std[_idx]
            mean = torch.as_tensor(mean, dtype=torch.float32)
            std = torch.as_tensor(std, dtype=torch.float32)
            img  = Image.open(self.img_paths[_idx]).convert('RGB')
            img = self.transform(img)
            return {'mean':mean, 'std':std, 'image':img}
        else:
            img  = Image.open(self.img_paths[_idx]).convert('RGB')

        return {'image':self.transform(img)}
        # return {'image':self.transform(img), 'image_vit':self.transform_vit(img)}

# DATAset for BDD100k dataset
class BDD100k_clear_daytime(BDD100k):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.transform_img = self.transform
        self.transform_img_vit = self.transform_vit
        self.load_npy=load_npy

        # self.img_paths = sorted([x for x in Path(os.path.join(root,f'images/100k/{split}')).glob('*.jpg')])
        self.img_paths = self.load_img_paths(f'{self.root_dir}/labels/{split}_clear_daytime_images.txt')
        print(f'length of the dataset : {len(self.img_paths)}')

        if dset_size is None:
            self.dset_size = len(self.img_paths)
        else:
            self.dset_size = dset_size

    def load_img_paths(self, file_path):
        # Load image paths from a text file
        with open(file_path, 'r') as file:
            # img_paths = [line.strip().replace('data1','scratch/slurm-user10-nims/data_whie') for line in file.readlines()]
            img_paths = [os.path.join(f'{self.root_dir}/images/100k/{self.split}',line.strip()) for line in file.readlines()]
        return img_paths






class ClevrTex(Dataset):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.full_dataset_path = os.path.join(root_dir, 'clevrtex-full.hdf5')
        self.split = split
        self.img_size = img_size

        self.dataset, _ = self._load_data()
        self.data = {}
        self.transform = self.get_transform()

        # TODO : val split
        if split == 'train':
            self.len_total_imgs = 40000
        else:
            self.len_total_imgs = 1000
        self.data['image'] = self.dataset['image'][:self.len_total_imgs]
        print(f'length of the dataset : {self.len_total_imgs}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 1000
        else:
            self.dset_size = dset_size


    def _load_data(self):
        """Loads data and metadata.

        By default, the data is a dict with h5py.Dataset values, but when overriding
        this method we allow arrays too."""
        return _load_data_hdf5(data_path=self.full_dataset_path)

    def __len__(self):
        # return self.len_total_imgs
        return self.dset_size

    def _preprocess_feature(self, feature: np.ndarray, feature_name: str) -> Any:
        """Preprocesses a dataset feature at the beginning of `__getitem__()`.

        Args:
            feature: Feature data.
            feature_name: Feature name.

        Returns:
            The preprocessed feature data.
        """
        # if feature_name == "image":
        #     return (
        #         torch.as_tensor(feature, dtype=torch.float32).permute(2, 0, 1)
        #     )
        return feature

    def __getitem__(self, idx):
        _idx = random.randint(0, self.len_total_imgs-1) 
        out = {}
        out['image'] = self.transform(
                self._preprocess_feature(
                    self.data['image'][_idx], 'image')
        )
        return out

    def get_transform(self):
        if self.split=='train':
            # resize to 64x64
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.5], [0.5]),
            ])
        return augmentations

class Clevr(ClevrTex):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.full_dataset_path = os.path.join(root_dir, 'clevr_10-full.hdf5')
        # self.full_dataset_path = os.path.join(root_dir, 'clevr_10-full_new.hdf5')
        self.split = split
        self.img_size = img_size

        self.dataset, _ = self._load_data()
        self.data = {}
        self.transform = self.get_transform()

        # TODO : val split
        if split == 'train':
            # test with only 10000 images
            self.len_total_imgs = 90000
        else:
            self.len_total_imgs = 1000
        self.data['image'] = self.dataset['image'][:self.len_total_imgs]
        print(f'length of the dataset : {self.len_total_imgs}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs
        else:
            self.dset_size = dset_size
            

class MULTIDSPRITE(ClevrTex):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.full_dataset_path = os.path.join(root_dir, 'multidsprites_colored_on_grayscale-full.hdf5')
        self.split = split
        self.img_size = img_size

        self.dataset, _ = self._load_data()
        self.data = {}
        self.transform = self.get_transform()

        # TODO : val split
        self.len_total_imgs = 90000

        self.data['image'] = self.dataset['image'][:self.len_total_imgs]
        print(f'length of the dataset : {self.len_total_imgs}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 1000
        else:
            self.dset_size = dset_size

    def get_transform(self):
        if self.split=='train':
            # resize to 64x64
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize(self.img_size, antialias=True),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize(self.img_size, antialias=True),
                transforms.Normalize([0.5], [0.5]),
            ])
        return augmentations


    def __getitem__(self, idx):
        _idx = random.randint(0, self.len_total_imgs-1) 
        out = {}
        out['image'] = self.transform(
                self._preprocess_feature(
                    self.data['image'][_idx], 'image')
        )
        return out


class COMB_DSPRITES(Dataset):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.split=split
        # self.root = os.path.join(root_dir, split)
        self.root = os.path.join(root_dir)
        self.img_size = img_size
        self.num_segs = 5
        self.transform = self.get_transform()

        self.total_imgs = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_images_simple_rand4.npz'))['images']
        self.total_masks = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_masks_simple_rand4.npz'))['masks']

        print(f'length of the dataset : {len(self.total_imgs)}')
        if dset_size is None:
            self.dset_size = len(self.total_imgs)
        else:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size
        # return len(self.total_imgs)

    def get_transform(self):
        if self.split=='train':
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            augmentations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        return augmentations

    def __getitem__(self, idx):
        _idx = random.randint(0, len(self.total_imgs)-1) 
        img = self.transform(self.total_imgs[_idx])
        all_mask = self.total_masks[_idx]

        #  # random horizontal flip the img
        #if random.random() > 0.5:
        #    img = img.flip(-1)

        #    # also flip numpy mask array
        #    all_mask = np.flip(all_mask, axis=1).copy()

        # load mask
        mask = []
        all_mask = torch.as_tensor(all_mask, dtype=torch.uint8)
        all_mask = rearrange(all_mask, 'h w c -> c h w')
        for i in range(self.num_segs):
            _mask = all_mask==i
            _mask = _mask.squeeze(0)
            mask.append(_mask.long())

        # stack mask : shape of mask (num_segs, height, width)
        mask = torch.stack(mask, dim=0)

        is_foreground = (mask.sum(-1).sum(-1) > 0).long().unsqueeze(-1)
        is_foreground[0] = 0
        is_background = 1-is_foreground
        visibility = is_foreground
        is_modified = torch.zeros_like(visibility).squeeze(-1)

        # align the format : shape is now become (num_segs, 1, height, width)
        mask = mask.unsqueeze(1).long()
        return {'image':img, 'mask':mask, 'is_foreground':is_foreground,
                'is_background':is_background,
                'is_modified':is_modified,
                'visibility':visibility,
                }


class COMB_DSPRITES_STYLE(COMB_DSPRITES):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.split=split
        self.root = os.path.join(root_dir)
        self.img_size = img_size
        self.num_segs = 5
        self.transform = self.get_transform()

        self.total_imgs = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_images_global_styles3_rand4.npz'))['images']
        self.total_masks = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_masks_global_styles3_rand4.npz'))['masks']

        print(f'length of the dataset : {len(self.total_imgs)}')
        if dset_size is None:
            self.dset_size = len(self.total_imgs)
        else:
            self.dset_size = dset_size

        self.__getitem__(0)


class COMB_DSPRITES_TRANSFORM(COMB_DSPRITES):
    def __init__(self, root_dir, split, img_size, dset_size=None, seed=None, precision='fp32'):
        self.split=split
        self.root = os.path.join(root_dir)
        self.img_size = img_size
        self.num_segs = 5
        self.transform = self.get_transform()

        self.total_imgs = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_images_global_tf3_rand4.npz'))['images']
        self.total_masks = np.load(os.path.join(self.root, 'prelim/multi_dsprites/train_masks_global_tf3_rand4.npz'))['masks']

        print(f'length of the dataset : {len(self.total_imgs)}')
        if dset_size is None:
            self.dset_size = len(self.total_imgs)
        else:
            self.dset_size = dset_size

        self.__getitem__(0)


class Cars3D(Dataset):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.load_npy=load_npy

        self.total_imgs = np.load(os.path.join(root_dir,'cars3d-x64.npz'))['imgs']
        print(f'length of the dataset : {self.total_imgs.shape[0]}')

        if dset_size is None:
            self.dset_size = self.total_imgs.shape[0]
        else:
            self.dset_size = dset_size
        self.__getitem__(0)

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, self.total_imgs.shape[0]-1)

        if self.load_npy:
            mean = self.latent_mean[_idx]
            std = self.latent_std[_idx]
            mean = torch.as_tensor(mean, dtype=torch.float32)
            std = torch.as_tensor(std, dtype=torch.float32)
            img  = Image.open(self.img_paths[_idx]).convert('RGB')
            img = self.transform(img)
            return {'mean':mean, 'std':std, 'image':img}
        else:
            img = self.total_imgs[_idx] 

        return {'image':self.transform(img)}

class Shapes3D(Dataset):
    def __init__(self, root_dir, img_size, dset_size=None, split='train', seed=None, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size

        with h5py.File(os.path.join(root_dir, '3dshapes.h5')) as f: 
            self.total_imgs = f['images'][:]
            self.total_labels = f['labels'][:]
        print(f'length of the dataset : {len(self.total_imgs)}')

        if dset_size is None:
            self.dset_size = len(self.total_imgs)
        else:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, len(self.total_imgs)-1)
        img = self.total_imgs[_idx] 
        return {'image':self.transform(img)}


class MPI3D(Dataset):
    def __init__(self, root_dir, img_size, dset_size=None, split='train', seed=None, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size

        self.len_total_imgs = 1036800

        print(f'length of the dataset : {self.len_total_imgs}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 5000

        else:
            self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, self.len_total_imgs-1)

        img = Image.open(f'{self.root_dir}/{_idx}.jpg').convert('RGB')

        return {'image':self.transform(img)}


class LSUN_CHURCH64(Dataset):
    def __init__(self, root_dir, img_size, dset_size=None, split='train', seed=None, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size

        self.total_imgs = np.load(os.path.join(root_dir, 'church_outdoor_train_lmdb_color_64.npy'))
        print(f'length of the dataset : {len(self.total_imgs)}')

        if dset_size is None:
            self.dset_size = len(self.total_imgs)
        else:
            self.dset_size = dset_size

        self.__getitem__(0)

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):

        _idx = random.randint(0, len(self.total_imgs)-1)
        img = self.total_imgs[_idx] 
        return {'image':self.transform(img)}

class LSUN_CHURCH256(Dataset):
    def __init__(self, root_dir, img_size, dset_size=None, split='train', seed=None, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size

        self.total_imgs = [x for x in Path(self.root_dir).glob('train/png/*.png')]
        print(f'length of the dataset : {len(self.total_imgs)}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 1000
        else:
            self.dset_size = dset_size

        self.__getitem__(0)

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):
        _idx = random.randint(0, len(self.total_imgs)-1)
        img = Image.open(self.total_imgs[_idx]).convert('RGB')
        return {'image':self.transform(img)}



class MSN(Dataset):
    def __init__(self, root_dir, img_size, dset_size=None, split='train', seed=None, precision='fp32'):
        self.split=split
        self.root_dir = os.path.join(root_dir, 'train')
        self.img_size = img_size

        self.total_imgs = [x for x in Path(self.root_dir).glob('*/*.png') if 'image' in x.name]
        print(f'length of the dataset : {len(self.total_imgs)}')

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 1000

        else:
            self.dset_size = dset_size

        self.__getitem__(0)

    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):
        _idx = random.randint(0, len(self.total_imgs)-1)
        img = Image.open(self.total_imgs[_idx]).convert('RGB')
        return {'image':self.transform(img)}



# class MSN_Easy(Dataset):
#     def __init__(self, root, img_size, num_segs, transform, transform_mask=None,
#             split='train', mode='train'):
#         self.split=split
#         self.root = os.path.join(root, split)
#         print(self.root)

#         self.img_size = img_size
#         self.mode = mode
#         self.transform = transform

#         self.total_imgs = [x for x in Path(self.root).glob('*/*.png') if 'image' in x.name]

#         self.total_imgs = sorted(self.total_imgs)

#         self.num_segs = num_segs
#         print(f'length of the dataset : {len(self.total_imgs)}')
#         # self.__getitem__(0)

#     def __len__(self):
#         return len(self.total_imgs)

#     def __getitem__(self, idx):
#         img = Image.open(self.total_imgs[idx]).convert('RGB')
#         img = self.transform(img)

#         if self.split=='train':
#             # random rgb augmentation
#             # if random.random() > 0.5:
#             #     img = img/2 + 0.5
#             #     img = img * torch.tensor((1.5, 1.0, 0.5))[:, None, None]
#             #     img = img.clip(0,1)
#             #     img = 2*img - 1.0
#             return {'image':img}

#         elif self.mode == 'validation':
#             # load mask
#             masks_path = str(self.total_imgs[idx]).replace('image', 'mask')
#             masks_path = masks_path.replace('png', 'npy')
#             masks = np.load(masks_path)

#             mask = []
#             for i in range(self.num_segs):
#                 # shape of mask : (height, width)
#                 # normalize to one-hot
#                 _mask = torch.as_tensor(masks==(i+1), dtype=torch.long).squeeze(-1)

#                 # resize to 128x128 masks
#                 resized_mask = torch.nn.functional.interpolate(_mask[:, 40:-40].float().unsqueeze(0).unsqueeze(0), size=(128,128)).long().squeeze()

#                 mask.append(resized_mask)

#             # stack mask : shape of mask (num_segs, height, width)
#             mask = torch.stack(mask, dim=0)

#             is_foreground = (mask.sum(-1).sum(-1) > 0).long().unsqueeze(-1)
#             is_foreground[0] = 0
#             is_background = 1-is_foreground
#             visibility = is_foreground
#             is_modified = torch.zeros_like(visibility).squeeze(-1)


#             # align the format : shape is now become (num_segs, 1, height, width)
#             mask = mask.unsqueeze(1)
#             return {'image':img, 'mask':mask, 'is_foreground':is_foreground,
#                     'is_background':is_background,
#                     'is_modified':is_modified,
#                     'visibility':visibility,
#                     }

class AAHQ_and_FFHQ_256(Dataset):
    def __init__(self, root_dir, img_size, vit_size=None, dset_size=None, split='train', seed=None, load_npy=False, precision='fp32'):
        self.split=split
        self.root_dir = root_dir
        self.img_size = img_size
        self.vit_size = vit_size
        self.load_npy=load_npy

        self.img_paths = sorted([x for x in Path(root_dir).glob('*.png')])
        print(f'length of the dataset : {len(self.img_paths)}')

        # if dset_size is None:
        #     self.dset_size = len(self.img_paths)
        # else:
        #     self.dset_size = dset_size

        if dset_size is None:
            self.dset_size = self.len_total_imgs if split=='train' else 1000

        else:
            self.dset_size = dset_size



    def __len__(self):
        return self.dset_size

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(sample)

    def __getitem__(self, idx):
        _idx = random.randint(0, len(self.img_paths)-1)

        if self.load_npy:
            raise NotImplementedError
        else:
            img  = Image.open(self.img_paths[_idx]).convert('RGB')

        return {'image':self.transform(img)}


class ClevrStyle(Dataset):
    def __init__(self, root_dir, img_size=(128, 128), dset_size=None, split='train', seed=None, precision='fp32'):
        self.split = split
        if split=='train':
            self.root_dir = os.path.join(root_dir, split)
        elif split=='test':
            self.root_dir = os.path.join(root_dir, split, 'images')
            print(self.root_dir)
        self.img_size = img_size

        if 'clevr-easy' in root_dir:
            self.num_attributes=3
        elif 'clevr-hard' in root_dir:
            self.num_attributes=3 # TODO we should check it 
        elif 'clevrtex-large' in root_dir:
            self.num_attributes=3 # TODO we should check it 
        elif 'clevr_stylized' in root_dir:
            self.num_attributes=3 # TODO we should check it
        else:
            raise NotImplementedError
        
        # load data
        if split == 'train':
            data_dir = self._data_pkl_path(self.root_dir)
            self.from_pkl = os.path.exists(data_dir)
            self.from_pkl = False
            if self.from_pkl:            
                # load vae encoded data
                with open(data_dir, 'rb') as f:
                    self.img_feats = torch.from_numpy(pickle.load(f)['feats'])
                
                self.total_imgs = self.img_feats.shape[0]
                
                # print
                print(f'load from data.pkl')
            
            else:
                # load images
                self.img_paths = sorted([x for x in Path(self.root_dir).glob('*.png') if self._check(x)])
                self.total_imgs = len(self.img_paths)
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.Normalize(mean=0.5, std=0.5),
                ])
        elif split == 'test':
            self.from_pkl = None
            # load images
            self.img_paths = sorted([x for x in Path(self.root_dir).glob('*.png') if 'mask' not in x.name])
            self.total_imgs = len(self.img_paths)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=0.5, std=0.5),
            ])
            
        print(f'length of the dataset : {self.total_imgs}')
        
        # dset size
        self.dset_size = dset_size if dset_size is not None else (self.total_imgs if split == 'train' else 1000)

    
    def _data_pkl_path(self, root_dir):
        return Path(root_dir).parent.joinpath('data.pkl')

    def _check(self, x): return True

    def __len__(self): return self.dset_size

    def __getitem__(self, idx):
        if self.split == 'train':
            _idx = random.randint(0, self.total_imgs - 1)
            img = self.img_feats[_idx] if self.from_pkl else self.transform(Image.open(self.img_paths[_idx]).convert('RGB'))
            return {'image': img}
        else:
            _idx = idx
            img = self.transform(Image.open(self.img_paths[_idx]).convert('RGB'))
            _mask_path = str(self.img_paths[_idx]).replace('.png', '_mask.png')
            _json_path = str(self.img_paths[_idx]).replace('.png', '.json').replace('images', 'scenes')
            
            # codes
            color_codes = {
                "gray": 0,
                "red": 1,
                "blue": 2,
                "green": 3,
                "brown": 4,
                "purple": 5,
                "cyan": 6,
                "yellow": 7,
            }
            shape_codes = {
                "cube": 0,
                "sphere": 1,
                "cylinder": 2,
            }
            
            # mask colors
            object_mask_colors = torch.Tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ])  # N, C
            max_num_objs = object_mask_colors.shape[0]

            eps = 0.001

            # masks
            mask_image = Image.open(_mask_path).convert("RGB")
            # mask_image = mask_image.resize((self.img_size, self.img_size))
            mask_image = transforms.ToTensor()(mask_image)  # C, H, W
            masks = (mask_image[None, :, :, :] < object_mask_colors[:, :, None, None] + eps) & \
                    (mask_image[None, :, :, :] > object_mask_colors[:, :, None, None] - eps)
            masks = masks.float().prod(1, keepdim=True)  # N, 1, H, W

            masks = torch.cat([torch.ones_like(masks[:1])*0.1, masks], dim=0) # add background mask
            masks = masks.argmax(0)


            # annotations
            annotations = torch.zeros(max_num_objs, self.num_attributes)  # N, G
            # _annotations = torch.zeros(max_num_objs, self.num_attributes)  # N, G

            # with open(_json_path) as f:
            #     data = json.load(f)
            #     object_list = data["objects"]
            #     for i, object in enumerate(object_list):
            #         # shape
            #         _annotations[i, 0] = shape_codes[object["shape"]]

            #         # color
            #         _annotations[i, 1] = color_codes[object["color"]]

            #         # position
            #         K = 3
            #         _annotations[i, 2] = np.digitize(object['3d_coords'][0], np.linspace(-4 - eps, 4 + eps, K + 1)) - 1
            #         _annotations[i, 2] = annotations[i, 2] * K + np.digitize(object['3d_coords'][1], np.linspace(-3 - eps, 4 + eps, K + 1)) - 1

            return {'image': img,  # C, H, W
                    'mask': masks,  # N, 1, H, W
                    'annotation': annotations,
                    'json_path': _json_path}  # N, G


class ClevrTexLarge(ClevrStyle):
    def _data_pkl_path(self, root_dir):
        return Path(root_dir).joinpath('data.pkl')
    
    def _check(self, x):
        return str(x).split('_')[-2].lower() == 'large'


class ClevrStyle2(ClevrStyle):
    def _data_pkl_path(self, root_dir):
        return Path(root_dir).parent.joinpath('data2.pkl')

    def _check(self, x):
        return 'draw' not in str(x)
