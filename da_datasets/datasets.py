#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import os
from utils import get_data, dir_to_list, get_n_shot, get_n_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ImgDataset(Dataset):
    """ Domain Dataset
    Parameters
    ----------
    x: list
        list of paths to images
    y: list
        list of labels
    transform: callable
        transform to apply to image
    """
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.transform(Image.open(self.x[index])), self.y[index]


class Office31():
    """ Office-31 Dataset

    Data structure:
        <root>/<dir_data>/<domain_name>/<class_name>

    Parameters
    ----------
    root: str
        root directory of stored data
    download: bool
        if True:
            download data if not already there, otherwise load
        if False:
            don't download even if data is not present
    dir_data: str, optional
        name of directory where data is stored, defaults to self.name
    """

    name = 'office-31'
    domains = ['amazon', 'dslr', 'webcam']
    splits = ['full', 'n-shot', 'list']
    urls = {
            'amazon.tar.gz': 'https://mega.nz/file/pnB3FJAa#4LqROlNwBmz0uLVV_6Ud2PkUw_urNjW4KyUXZ59qAgw',
            'dslr.tar.gz': 'https://mega.nz/file/4qwjlL6L#HXYoGVpHH8QkBIPz8jygtPSQRYjRY4a_Ftmp6UPQjYc',
            'webcam.tar.gz': 'https://mega.nz/file/ZnYyQLiZ#6zt4MxA4uExPRFfW9kWb8ayX8mw8VhZ9-VyjKPeTHMk',
            }
    c2i = {
           'back_pack': 0,
           'bike': 1,
           'bike_helmet': 2,
           'bookcase': 3,
           'bottle': 4,
           'calculator': 5,
           'desk_chair': 6,
           'desk_lamp': 7,
           'desktop_computer': 8,
           'file_cabinet': 9,
           'headphones': 10,
           'keyboard': 11,
           'laptop_computer': 12,
           'letter_tray': 13,
           'mobile_phone': 14,
           'monitor': 15,
           'mouse': 16,
           'mug': 17,
           'paper_notebook': 18,
           'pen': 19,
           'phone': 20,
           'printer': 21,
           'projector': 22,
           'punchers': 23,
           'ring_binder': 24,
           'ruler': 25,
           'scissors': 26,
           'speaker': 27,
           'stapler': 28,
           'tape_dispenser': 29,
           'trash_can': 30,
            }

    def __init__(self, root, download=True, dir_data=None,
                 transform=None):

        super().__init__()
        self.root = root
        self.download = download
        self.dir_data = self.name if dir_data is None else dir_data

        # full path to data
        self.full_path = os.path.join(self.root, self.dir_data)
        # check if data is already there
        if not os.path.exists(self.full_path):
            print("Data not found.")
            if download:
                print("Downloading.")
                self._get_data()
            else:
                print("Download the dataset to continue.")
        else:
            # check if domains
            domains_exist = True
            for domain in self.domains:
                if not os.path.exist(os.path.join(self.full_path, domain)):
                    domains_exist = False
                    break
            if domains_exist:
                print("Data already downloaded and unpacked.")
            else:
                archives_exist = True
                for archive in self.urls
                    if not os.path.exist(os.path.join(self.full_path, archive)):
                        archives_exist = False
                        break
                if archives_exist:
                    print("Data already downloaded. Unpacking.")
                    self._get_data()

    def _get_data(self):
        for name_archive, url in self.urls.items():
            print(f"Getting {name_archive}")
            # filename
            dir_unpack = name_archive.split('.')[0]
            # where to save archive file
            path_archive = os.path.join(self.full_path,
                                        f"{self.name}-{name_archive}")
            # where to unpack
            path_data = os.path.join(self.full_path, dir_unpack)
            # download and unpack
            get_data(url, path_archive, path_data)

    def _load_data(self, domain):
        if domain.lower() not in self.domains:
            print(f"No domain called {domain}.")
            return
        path_domain_data = os.path.join(self.full_path, domain)
        data = dir_to_list(path_domain_data)
        images = np.array([item[0] for item in data])  # image paths
        classes = np.array([item[1] for item in data])  # class names
        labels = np.array([self.c2i[c] for c in classes])  # label indices
        return images, labels

    def get_dataset(self, domain, split='full', transform=None):
        """ Get torch dataset

        Parameters
        ----------
        domain: str
            one of ['amazon', 'dslr', 'webcam']
        split: str or list
            if 'full':
                return all data
            if '<n>-shot' where <n> is an integer:
                return 2 datasets
                first one has exactly <n> examples per class
                second one has all examples that are not in first dataset
            if list:
                return L+1 datasets, where L is the length of the list
                dataset[i] contains split[i] * len(all_data) examples
                example: split = [0.7, 0.2]
                    first dataset contains 70% of data
                    2nd dataset contains 20% of data
                    third dataset contains 10% of data
        transform: callable, optional:
            data transformation to apply
        """

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ])

        images, labels = self._load_data(domain)
        if split == 'full':
            return ImgDataset(images, labels, transform)
        elif 'shot' in split:
            n = int(split.split('-')[0])
            index_1, index_2 = get_n_shot(labels, n)
            ds_1 = ImgDataset(images[index_1], labels[index_1], transform)
            ds_2 = ImgDataset(images[index_2], labels[index_2], transform)
            return ds_1, ds_2
        elif type(split) == list:
            indices = get_n_split(labels, split)
            datasets = []
            for index in indices:
                dataset = ImgDataset(images[index], labels[index], transform)
                datasets.append(dataset)
            return datasets


class OfficeHome():
    """ Office Home Dataset

    Data structure:
        <root>/<dir_data>/<domain_name>/<class_name>

    Parameters
    ----------
    root: str
        root directory of stored data
    download: bool
        if True:
            download data if not already there, otherwise load
        if False:
            don't download even if data is not present
    dir_data: str, optional
        name of directory where data is stored, defaults to self.name
    """

    name = 'office-home'
    domains = ['artistic', 'clip', 'product', 'real_world']
    splits = ['full', 'n-shot', 'list']
    urls = {
            'artistic.tar.gz': 'https://mega.nz/file/I3oQCbgZ#GxRJo32evBYBn5wuepfP2e1lT687boXi6iwP3yh4NDM',
            'clip_art.tar.gz': 'https://mega.nz/file/9mRjyawa#pEls_nOfoo0Nw_RrecV6L9rvEqroyOgrciuZnB0jExg',
            'product.tar.gz': 'https://mega.nz/file/Mn4jWKoK#7eWF43duQAdkRlGqEAs4m-oxiYmKHEaFucERomD5Ij0',
            'real_world.1.tar.gz': 'https://mega.nz/file/ImpEzTLD#O71gEPjNu5STgtJPdY389o6BVqO_EFojdkhPwGIYMHw',
            'real_world.2.tar.gz': 'https://mega.nz/file/Z65XnK5A#tkpwQ8SPZe4bXp1Q16lyZCIqCbebisEE76oPcBTXupQ',
            'real_world.3.tar.gz': 'https://mega.nz/file/4qwCQY7L#IvI_yJsh9mr7EezhxRCIIGDSTR6kUMek7F57e0xHU5w',
            }
    c2i = {
        'alarm_clock': 0,
        'backpack': 1,
        'batteries': 2,
        'bed': 3,
        'bike': 4,
        'bottle': 5,
        'bucket': 6,
        'calculator': 7,
        'calendar': 8,
        'candles': 9,
        'chair': 10,
        'clipboards': 11,
        'computer': 12,
        'couch': 13,
        'curtains': 14,
        'desk_lamp': 15,
        'drill': 16,
        'eraser': 17,
        'exit_sign': 18,
        'fan': 19,
        'file_cabinet': 20,
        'flipflops': 21,
        'flowers': 22,
        'folder': 23,
        'fork': 24,
        'glasses': 25,
        'hammer': 26,
        'helmet': 27,
        'kettle': 28,
        'keyboard': 29,
        'knives': 30,
        'lamp_shade': 31,
        'laptop': 32,
        'marker': 33,
        'monitor': 34,
        'mop': 35,
        'mouse': 36,
        'mug': 37,
        'notebook': 38,
        'oven': 39,
        'pan': 40,
        'paper_clip': 41,
        'pen': 42,
        'pencil': 43,
        'postit_notes': 44,
        'printer': 45,
        'push_pin': 46,
        'radio': 47,
        'refrigerator': 48,
        'ruler': 49,
        'scissors': 50,
        'screwdriver': 51,
        'shelf': 52,
        'sink': 53,
        'sneakers': 54,
        'soda': 55,
        'speaker': 56,
        'spoon': 57,
        'table': 58,
        'telephone': 59,
        'toothbrush': 60,
        'toys': 61,
        'trash_can': 62,
        'tv': 63,
        'webcam': 64,
            }

    def __init__(self, root, download=True, dir_data=None,
                 transform=None):
        super().__init__()
        self.root = root
        self.download = download
        self.dir_data = self.name if dir_data is None else dir_data

        # full path to data
        self.full_path = os.path.join(self.root, self.dir_data)
        # check if data is already there
        if not os.path.exists(self.full_path):
            print("Data not found.")
            if download:
                print("Downloading.")
                self._get_data()
            else:
                print("Download the dataset to continue.")
        else:
            # check if domains
            domains_exist = True
            for domain in self.domains:
                if not os.path.exist(os.path.join(self.full_path, domain)):
                    domains_exist = False
                    break
            if domains_exist:
                print("Data already downloaded and unpacked.")
            else:
                archives_exist = True
                for archive in self.urls
                    if not os.path.exist(os.path.join(self.full_path, archive)):
                        archives_exist = False
                        break
                if archives_exist:
                    print("Data already downloaded. Unpacking.")
                    self._get_data()

    def _get_data(self):
        for name_archive, url in self.urls.items():
            print(f"Getting {name_archive}")
            # filename
            dir_unpack = name_archive.split('.')[0]
            # where to save archive file
            path_archive = os.path.join(self.full_path,
                                        f"{self.name}-{name_archive}")
            # where to unpack
            path_data = os.path.join(self.full_path, dir_unpack)
            # download and unpack
            get_data(url, path_archive, path_data)

    def _load_data(self, domain):
        if domain.lower() not in self.domains:
            print(f"No domain called {domain}.")
            return
        path_domain_data = os.path.join(self.full_path, domain)
        data = dir_to_list(path_domain_data)
        images = np.array([item[0] for item in data])  # image paths
        classes = np.array([item[1] for item in data])  # class names
        labels = np.array([self.c2i[c] for c in classes])  # label indices
        return images, labels

    def get_dataset(self, domain, split='full', transform=None):
        """ Get torch dataset

        Parameters
        ----------
        domain: str
            one of ['artistic', 'clip_art', 'product', 'real_world']
        split: str or list
            if 'full':
                return all data
            if '<n>-shot' where <n> is an integer:
                return 2 datasets
                first one has exactly <n> examples per class
                second one has all examples that are not in first dataset
            if list:
                return L+1 datasets, where L is the length of the list
                dataset[i] contains split[i] * len(all_data) examples
                example: split = [0.7, 0.2]
                    first dataset contains 70% of data
                    2nd dataset contains 20% of data
                    third dataset contains 10% of data
        transform: callable, optional:
            data transformation to apply
        """

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ])

        images, labels = self._load_data(domain)
        if split == 'full':
            return ImgDataset(images, labels, transform)
        elif 'shot' in split:
            n = int(split.split('-')[0])
            index_1, index_2 = get_n_shot(labels, n)
            ds_1 = ImgDataset(images[index_1], labels[index_1], transform)
            ds_2 = ImgDataset(images[index_2], labels[index_2], transform)
            return ds_1, ds_2
        elif type(split) == list:
            indices = get_n_split(labels, split)
            datasets = []
            for index in indices:
                dataset = ImgDataset(images[index], labels[index], transform)
                datasets.append(dataset)
            return datasets
