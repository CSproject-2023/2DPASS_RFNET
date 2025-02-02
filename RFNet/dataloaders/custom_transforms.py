import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        img = np.array(img).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        # mean and std for original depth images
        mean_depth = 0.12176
        std_depth = 0.09752

        depth /= 255.0
        depth -= mean_depth
        depth /= std_depth

        return {'image': img,
                'depth': depth}


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        depth = sample['depth']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(depth).astype(np.float32)

        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()

        return {'image': img,
                'depth': depth}

class CropBlackArea(object):
    """
    crop black area for depth image
    """
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        width, height = img.size
        left = 140
        top = 30
        right = 2030
        bottom = 900
        # crop
        img = img.crop((left, top, right, bottom))
        depth = depth.crop((left, top, right, bottom))
        # resize
        img = img.resize((width,height), Image.BILINEAR)
        depth = depth.resize((width,height), Image.BILINEAR)
        # img = img.resize((512,1024), Image.BILINEAR)
        # depth = depth.resize((512,1024), Image.BILINEAR)
        # mask = mask.resize((512,1024), Image.NEAREST)

        return {'image': img,
                'depth': depth}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        depth = depth.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'depth': depth}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'depth': depth}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)  # depth多余的部分填0
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']

        assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'depth': depth}

class Relabel(object):
    def __init__(self, olabel, nlabel):  # change trainid label from olabel to nlabel
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        # assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor,
        #                                                            torch.ByteTensor)), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor