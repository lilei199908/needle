from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR CODE
        import gzip
        import struct
        with gzip.open(image_filename, 'rb') as f:
            image_bytes = f.read()
            magic_number, num_images, rows, cols = struct.unpack(">iiii", image_bytes[:16])
            pixels = struct.unpack("%dB" % (num_images * rows * cols), image_bytes[16:])
            self.pixels = np.array(pixels, dtype=np.float32).reshape(num_images,  rows * cols) / 255
        
        with gzip.open(label_filename, 'rb') as f:
            label_bytes = f.read()
            magic_number, num_labels = struct.unpack(">ii", label_bytes[:8])
            labels = struct.unpack("%dB" % num_labels, label_bytes[8:])
            self.labels = np.array(labels, dtype=np.uint8)
            
        self.transforms = transforms
        ### END YOUR CODE

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.pixels[index], self.labels[index]
        if self.transforms:
            X_in = X.reshape((28, 28, -1))
            X_out = self.apply_transforms(X_in)
            return X_out.reshape(-1, 28 * 28), y
        else:
            return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        assert self.pixels.shape[0] == self.labels.shape[0]
        return self.pixels.shape[0]
        ### END YOUR SOLUTION
        
        
        
    
    