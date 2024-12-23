import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, get_worker_info

# Function to preprocess and save the dataset in HDF5
def preprocess_and_save_to_hdf5(dataset, hdf5_path):
    print("Preprocessing dataset and saving to HDF5...")
    with h5py.File(hdf5_path, "w") as hdf5_file: #crearea unui fișier HDF5 gol
        # Dimensiuni - (număr total imagini, canale RGB, înălțime, lățime)
        img_shape = (len(dataset), 1, 64, 64)
        # Imaginile sunt stocate ca valori `float32`
        # utile pentru operații ulterioare precum normalizarea
        hdf5_file.create_dataset("images", shape=img_shape, dtype=np.float32)

        # Preprocesare și salvare fiecare imagine
        for idx, (img, _) in enumerate(dataset):
            hdf5_file["images"][idx] = img.numpy()
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(dataset)} images")

    print(f"Dataset saved to {hdf5_path}")

# Custom Dataset to load from HDF5
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.dataset_length = self._get_dataset_length()

    def _get_dataset_length(self):
        # Open the file temporarily to determine dataset size
        with h5py.File(self.hdf5_path, "r") as file:
            return len(file["images"])

    def _ensure_open(self):
        if self.file is None:
            worker_info = get_worker_info()
            if worker_info is None:  # Single-process DataLoader
                self.file = h5py.File(self.hdf5_path, "r")
            else:  # Multi-process DataLoader
                self.file = h5py.File(self.hdf5_path, "r", libver='latest')

    def __getitem__(self, idx):
        # Open the HDF5 file for each worker
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            # Access the image at the given index
            image = hdf5_file["images"][idx]
            return torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return self.dataset_length
