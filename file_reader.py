def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as label_path:
        labels = np.frombuffer(label_path.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as img_path:
        images = np.frombuffer(img_path.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
