from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torchvision import datasets


class ImageHandler:

    def __init__(self, image_size, content_image_path, style_image_path, device, preserve_colors=False):
        self.image_size = image_size
        self.device = device
        get_color_from_image_path = content_image_path if preserve_colors else None

        self.content_image = self.image_loader(content_image_path)
        self.style_image = self.image_loader(style_image_path, get_color_from_image_path)
        assert self.content_image.size() == self.style_image.size(), "The content and style image must be of the same size"

    def image_loader(self, image_path, get_color_from_image_path=None):
        loader = transforms.Compose([
            transforms.Resize(self.image_size),
            MatchColorHistogram(get_color_from_image_path),
            transforms.ToTensor()
        ])
        image = Image.open(image_path)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def image_unloader(self, tensor):
        unloader = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        return unloader(image)

    def imshow(self, tensor, title=None):
        image = self.image_unloader(tensor)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.show()


class MatchColorHistogram(object):

    def __init__(self, color_from_image_path):
        self.color_from_image_path = color_from_image_path

    def __call__(self, image):
        if self.color_from_image_path is None:
            return image
        source_image = np.asarray(Image.open(self.color_from_image_path))/255.
        target_image = np.asarray(image)/255.

        mu_target = target_image.mean(axis=(0, 1))
        t = target_image - mu_target
        t = t.transpose(2, 0, 1).reshape(3, -1)
        sigma_target = t.dot(t.T) / t.shape[1]

        mu_source = source_image.mean(axis=(0, 1))
        s = source_image - mu_source
        s = s.transpose(2, 0, 1).reshape(3, -1)
        sigma_source = s.dot(s.T) / s.shape[1]

        chol_t = np.linalg.cholesky(sigma_target)
        chol_s = np.linalg.cholesky(sigma_source)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)

        color_transferred_image = ts.reshape(*target_image.transpose(2, 0, 1).shape).transpose(1, 2, 0)
        color_transferred_image += mu_source
        color_transferred_image *= 255.
        color_transferred_image = np.clip(color_transferred_image, 0, 255)
        return Image.fromarray(color_transferred_image.astype('uint8'))


class TrainStyleImageHandler:

    def __init__(self, image_size, style_image_path, dataset_path, batch_size, device,):
        self.image_size = image_size
        self.device = device

        train_dataset = datasets.ImageFolder(dataset_path, self.loader)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)

        self.style_image = self.style_image_loader(style_image_path, batch_size)

    @property
    def loader(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])

    def style_image_loader(self, style_image_path, batch_size):
        style_image = Image.open(style_image_path)
        style_image = self.loader(style_image)
        style_image = style_image.repeat(batch_size, 1, 1, 1)
        return style_image.to(self.device, torch.float)

