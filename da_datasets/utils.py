import os
import numpy as np
from mega import Mega
import tarfile
from zipfile import ZipFile
import shutil
from pathlib import Path

def get_archive_content(path_archive):
    """
    returns how top level directory in archive is called
    """
    if path_archive.endswith(('.gz', '.tar')):
        file = tarfile.open(path_archive).getmembers()[0].get_info()['name']
        if file.startswith('./'):
            file = file[2:]
        name_old = file.split('/')[0]
    elif path_archive.endswith('.zip'):
        with ZipFile(path_archive, 'r') as f:
            for f in f.namelist():
                name_old = f.split('/')[0]
                break
    return name_old


def get_data(url, path_archive, path_data):
    """ Download and unpack data from url
    url is downloaded into <root>
    data is unpacked into <root>/<dir_name>

    Parameters
    ----------
    url: str
        url to download
    path_archive:
        location of archive file
    path_data:
        where to move data

    Example:
        archive.zip contains directory <abc> in the lowest level
        path_archive = 'dir1/dir2/archive.zip'
        path_data = 'dir3/dir4'

        archive will be downloaded to <path_archive>
        archive will be unpacked to <dir3>
        after unpacking, there is a directory <abc> in <dir3>
        <abc> will be renamed to <dir4>
    """
    # archive dir and name
    filename_archive = path_archive.split('/')[-1]
    dir_archive = '/'.join(path_archive.split('/')[:-1])
    # data dir and name
    dir_data = '/'.join(path_data.split('/')[:-1])

    # make root dir
    if not(os.path.exists(dir_archive)):
        os.makedirs(dir_archive, exist_ok=True)
    # make data dir
    if not(os.path.exists(path_data)):
        os.makedirs(path_data, exist_ok=True)
    # check if archive is already downloaded
    print(path_archive)
    if not os.path.exists(path_archive):
        # download
        print(f"Downloading {filename_archive}")
        if url.startswith('https://mega.nz'):
            m = Mega()
            m.download_url(url,
                           dest_path=dir_archive,
                           dest_filename=filename_archive)
    else:
        print("Archive is already downloaded")
    # find how unpacked dir is called
    name_old = get_archive_content(path_archive)
    # unpack
    shutil.unpack_archive(path_archive, dir_data)
    name_unpacked = os.path.join(dir_data, name_old)
    # move to target location
    os.rename(name_unpacked, path_data)


def dir_to_list(root, extensions=['png', 'jpg', 'jpeg']):
    """ Get list of all files in dir
    root: str
        path to root directory
    extensions: list or tuple
        list or tuple of img extensions

    assumed directory structure
    root
    ├── class01
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ...
    ├── class02
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ...
     ...
    """
    extensions = tuple(extensions)

    def get_group(filename):
        return str(filename).split('/')[-2]

    # find all images
    p = Path(root)
    all_files = [(str(file), get_group(file))
                 for file in p.rglob("*")
                 if str(file).lower().endswith(extensions)]

    return all_files


def get_n_shot(labels, n=1):
    """ Split data for n-shot learning

    Returns 2 arrays, index_1 and index_2.
    labels[index_1] contains <n> examples of each class
    labels[index_2] contains all examples not in label[index_1]

    Parameters
    ----------
    labels: list
        list of all labels
    n: int
        how many examples to keep
    """
    labels = np.array(labels)

    index_1 = []
    index_2 = []

    index_all = np.arange(len(labels))

    for val in np.unique(labels):
        index = index_all[labels == val]
        np.random.shuffle(index)
        index_1.append(index[:n])  # take n items
        index_2.append(index[n:])  # take all other items

    index_1 = np.concatenate(index_1)
    index_2 = np.concatenate(index_2)
    return index_1, index_2


def get_n_split(labels, sizes=[0.8]):
    """ Split data into n datasets

    Parameters
    ----------
    labels: list
        list of labels
    sizes: list
        list of target dataset sizes
    """
    n_datasets = len(sizes) + 1
    labels = np.array(labels)
    size_total = len(labels)

    indices = [[] for _ in range(n_datasets)]
    leftover = []

    # calculate target sizes
    target_sizes = [int(size_total * size) for size in sizes]
    target_sizes.append(size_total - sum(target_sizes))
    # make index
    index_all = np.arange(size_total)

    # put 1 instance of each class in each return dataset
    for val in np.unique(labels):
        # find all items of class <val>
        index = index_all[labels == val]
        # shuffle
        np.random.shuffle(index)
        # put 1 example to all return datasets
        for return_index in range(n_datasets):
            indices[return_index].append(index[return_index])
        # append leftover index
        leftover.append(index[n_datasets:])

    # make leftover into 1 array
    leftover = np.concatenate(leftover)
    np.random.shuffle(leftover)
    # fill the datasets according to their target sizes
    start = 0
    for i, target_size in enumerate(target_sizes):
        # calculate how many items more to add
        target_size = target_size - len(indices[i])
        # add items
        indices[i].extend(leftover[start:start+target_size])
        # move start counter
        start += target_size

    indices = [np.array(index) for index in indices]

    return indices
