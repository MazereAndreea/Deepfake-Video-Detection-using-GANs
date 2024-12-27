import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Function to preprocess and save the dataset in HDF5
def preprocess_and_save_to_hdf5(dataset, hdf5_path):
    print("Preprocessing dataset and saving to HDF5...")
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Create HDF5 datasets for images
        img_shape = (len(dataset), 3, 64, 64)  # Assuming RGB images resized to 64x64
        hdf5_file.create_dataset("images", shape=img_shape, dtype=np.float32)

        # Preprocess and store each image
        for idx, (img, _) in enumerate(dataset):
            hdf5_file["images"][idx] = img.numpy()
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(dataset)} images")

    print(f"Dataset saved to {hdf5_path}")

# Custom Dataset to load from HDF5
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.hdf5_file = None
        with h5py.File(self.hdf5_path, "r") as file:
            self.dataset_length = len(file["images"])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, "r")
        image = self.hdf5_file["images"][idx]
        return torch.tensor(image)
