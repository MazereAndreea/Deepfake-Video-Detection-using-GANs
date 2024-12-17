import os
from discriminator import Discriminator
# from displayImageFromVideo import display_image
# from displayImageFromVideo import display_images_from_video_list
# from displayImageFromVideo import play_video
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from HDF5Dataset import HDF5Dataset, preprocess_and_save_to_hdf5
import multiprocessing
import torch.distributed as dist

def sendData(smth=None):
    return smth

NOISE_DIM = 100
NUM_EPOCHS = 5
BATCH_SIZE = 256
HDF5_FILE = "preprocessed_dataset.h5"


def split_dataset(dataset, num_splits, rank):
    """
    Split a dataset into `num_splits` subsets and return the subset for the current rank.

    Parameters:
        dataset: Original dataset (HDF5Dataset or other PyTorch dataset).
        num_splits: Total number of processes (splits).
        rank: The current process rank (0-indexed).

    Returns:
        SubDataset for the specific rank.
    """
    total_len = len(dataset)
    indices = list(range(total_len))
    split_size = total_len // num_splits
    split_indices = indices[rank * split_size: (rank + 1) * split_size]

    # Handle remainder for uneven splits
    if rank == num_splits - 1:
        split_indices.extend(indices[(rank + 1) * split_size:])

    return Subset(dataset, split_indices)

def main(rank, num_processes):

    dist.init_process_group(backend='gloo', init_method='file:///tmp/sharedfile', rank=rank, world_size=num_processes)
    generator = Generator(NOISE_DIM)
    discriminator = Discriminator()

    device = 'cpu'
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    sendData(generator)

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # CelebA dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        transforms.CenterCrop(178),# Decuparea centrului pentru păstrarea feței.
        transforms.Resize(64), # Redimensionarea imaginilor la 64x64.
        transforms.ToTensor(), # Conversia imaginii la tensor PyTorch.
        # Normalizarea imaginilor la intervalul [-1, 1].
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale values to range [-1, 1]
    ])

    # Check if the HDF5 file exists
    if not os.path.exists(HDF5_FILE):
        # Preprocess dataset and save if not already cached
        if rank == 0:
            train_dataset = torchvision.datasets.CelebA(
                root='C:/Users/ANDREEA/PycharmProjects/Rn/data/',
                split='train',
                transform=transform,
                download=False
            )
            preprocess_and_save_to_hdf5(train_dataset, HDF5_FILE)
        # Barrier to ensure all ranks wait for preprocessing to complete
        dist.barrier()

    # Load data from HDF5
        # Load HDF5 dataset and split it for this worker
    full_dataset = HDF5Dataset(HDF5_FILE)
    worker_dataset = split_dataset(full_dataset, num_processes, rank)
    train_loader = DataLoader(worker_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # train_dataset = torchvision.datasets.CelebA(root='C:/Users/ANDREEA/PycharmProjects/Rn/data/', split = 'train', transform=transform, download=False)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    # Training loop

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):
            real_images = data
            real_images = real_images.to(device)

            # Train discriminator with real images
            discriminator_optimizer.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1, device=device)
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)
            real_loss.backward()

            # Train discriminator with fake images
            noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            fake_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            generator_optimizer.zero_grad()
            fake_labels = torch.ones(real_images.size(0), 1, device=device)
            fake_outputs = discriminator(fake_images)
            gen_loss = criterion(fake_outputs, fake_labels)
            gen_loss.backward()
            generator_optimizer.step()

            # Print losses
            if i % 100 == 0:
                print(f"Worker {rank + 1}, Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], "
                      f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, "
                      f"Generator Loss: {gen_loss.item():.4f}")

    # Generate and save images
    def generate_and_save_images(model, epoch, noise):
        model.eval()
        with torch.no_grad():
            fake_images = model(noise).cpu()
            fake_images = fake_images.view(fake_images.size(0), 3, 64, 64)
            for i in range(fake_images.size(0)):
                # Move channels to the last dimension for matplotlib
                image = fake_images[i].permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                image = (image * 0.5 + 0.5).numpy()  # Denormalize for proper visualization
                plt.subplot(4, 4, i+1)
                plt.imshow(image)
                plt.axis('off')

            plt.savefig(f'image_at_epoch_{epoch+1:04d}.png')
            plt.show()

    # Generate test noise
    test_noise = torch.randn(16, NOISE_DIM, device=device)
    generate_and_save_images(generator, NUM_EPOCHS, test_noise)
    dist.destroy_process_group()

if __name__ == "__main__":
    import os

    processes = []
    num_processes = multiprocessing.cpu_count()
    os.environ["C10D_DEBUG_RANDOM"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Localhost (use IP for distributed multi-node training)
    os.environ["MASTER_PORT"] = "29500"  # A random port number in the range 1024–65535 (avoid clashes)
    torch.multiprocessing.spawn(
        main,  # Target function
        args=(num_processes,),  # Arguments passed to `main`
        nprocs=num_processes,  # Total number of processes
        join=True  # Wait for processes to finish
    )

    print("All processes finished.")
    #main()
