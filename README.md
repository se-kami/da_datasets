## PyTorch datasets for domain adaptation
Datasets:
- Digits
- Image-Clef
- Natural Scene
- Office-31
- Office-Home
- VisDA2017

### General example
``` python
from da_datasets import datasets

dataset_maker = datasets.<DATASET_NAME>
pytorch_datasets = dataset_maker.get_dataset(domain=<DOMAIN_NAME>, split=<SPLIT>)
```

split can be:
- full - to get all examples
- n-shot - to get n examples per class
- list - to split into len(list) + 1 datasets
- train - for some datasets like MNIST that provide splits
- test - for some datasets like MNIST that provide splits

### Digits
This dataset contains 10 classes in 5 domains (MNIST, MNIST-M, SVHN, Synthetic, USPS).
To use it:
```python
digits = da_datasets.datasets.Digits(root='DATA', download=True)

mnist_train, mnist_test = digits.get_dataset(domain='mnist', split='1-shot')
```

### Image-Clef
This dataset contains 12 classes in 3 domains (Caltech-256 (C), ImageNet ILSVRC 2012 (I) and Pascal VOC 2012 (P))
To use it:
```python
image_clef = da_datasets.datasets.ImageClef(root='DATA', download=True)

p_train, p_test = digits.get_dataset(domain='p', split='1-shot')
```

### Natural Scene
This dataset contains 9 classes in 2 domains (CIFAR, STL)
To use it:
```python
natural = da_datasets.datasets.NaturalScene(root='DATA', download=True)

cifar_train, cifar_test = digits.get_dataset(domain='cifar', split='1-shot')
```

### Office-31
This dataset contains 31 classes in 3 domains(Amazon, DSLR, Webcam).
To use it:
```python
office_31 = da_datasets.datasets.Office31(root='DATA', download=True)

webcam_train, webcam_test = office_31.get_dataset(domain='webcam', split='1-shot')
```

### Office-Home
This dataset contains 65 classes in 4 domains(Artistic, Clip-Art, Product, Real World).
To use it:
```python
office_home = da_datasets.datasets.OfficeHome(root='DATA', download=True)

product_train, product_test = office_home.get_dataset(domain='product', split='1-shot')
```

### VisDA2017
This dataset contains 12 classes in 2 domains (Train, Test).
To use it:
```python
visda = da_datasets.datasets.VisDA2017(root='DATA', download=True)

visda_train, visda_test = digits.get_dataset(domain='train', split='1-shot')
```
