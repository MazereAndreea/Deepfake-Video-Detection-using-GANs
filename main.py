import os

from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile
from torchvision.utils import save_image

import discriminator
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
# import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from HDF5Dataset import HDF5Dataset, preprocess_and_save_to_hdf5
import multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from video_analysis import analyze_all_videos_in_directory
from torch.utils.tensorboard import SummaryWriter

NOISE_DIM = 100
NUM_EPOCHS = 10
BATCH_SIZE = 256
HDF5_FILE = "preprocessed_dataset.h5"
lr_gen = 0.005
lr_disc = 0.005

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


def train_step(generator, discriminator, data, real_labels, fake_labels, criterion, generator_optimizer,
               discriminator_optimizer, device):

    """
    Perform one step of training for the discriminator and generator.
    """
    real_images = data.to(device)
    real_outputs = discriminator(real_images)
    real_loss = criterion(real_outputs, real_labels)
    real_loss.backward()

    noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
    with torch.no_grad():
        fake_images = generator(noise)

    # Profile discriminator fake image processing
    fake_outputs = discriminator(fake_images.detach())
    fake_loss = criterion(fake_outputs, fake_labels)
    fake_loss.backward()
    discriminator_optimizer.step()

    # Profile generator step
    generator_optimizer.zero_grad()
    fake_outputs = discriminator(fake_images)
    gen_loss = criterion(fake_outputs, real_labels)  # Generator wants discriminator to think fake is real

    gen_loss.backward()
    generator_optimizer.step()

    return real_loss, fake_loss, gen_loss, fake_images

def main(rank, num_processes):

    try:
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(num_processes)
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29500',
            rank=rank,
            world_size=num_processes
        )

        device = torch.device("cpu")
        generator = Generator(NOISE_DIM).to(device)
        discriminator = Discriminator().to(device)

        generator = DDP(generator)
        discriminator = DDP(discriminator)

        generator_optimizer = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

        log_dir = './logs/tensorboard'
        writer = SummaryWriter(log_dir=log_dir)

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
                    root='/home/andreeamazere.am/C:/Users/ANDREEA/PycharmProjects/Rn/data/',
                    split='train',
                    transform=transform,
                    download=False
                )
                preprocess_and_save_to_hdf5(train_dataset, HDF5_FILE)
        dist.barrier()  # Synchronize all processes after preprocessing is completed by rank 0

        # Load data from HDF5
            # Load HDF5 dataset and split it for this worker
        full_dataset = HDF5Dataset(HDF5_FILE)
        limited_dataset = Subset(full_dataset, list(range(1000)))
        worker_dataset = split_dataset(limited_dataset, num_processes, rank)
        train_loader = DataLoader(
            worker_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # train_dataset = torchvision.datasets.CelebA(root='C:/Users/ANDREEA/PycharmProjects/Rn/data/', split = 'train', transform=transform, download=False)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        # Training loop

        # Training loop
        for epoch in range(NUM_EPOCHS):
            for i, data in enumerate(train_loader):
                real_labels = torch.ones(data.size(0), 1, device=device)
                fake_labels = torch.zeros(data.size(0), 1, device=device)

                # Perform training step and profile
                real_loss, fake_loss, gen_loss, fake_images = train_step(generator, discriminator, data,
                                                                         real_labels,
                                                                         fake_labels,
                                                                         criterion, generator_optimizer,
                                                                         discriminator_optimizer,
                                                                         device)

                if i % 100 == 0:
                    # Log losses to TensorBoard
                    writer.add_scalar('Loss/Discriminator', real_loss.item() + fake_loss.item(),
                                      epoch * len(train_loader) + i)
                    writer.add_scalar('Loss/Generator', gen_loss.item(), epoch * len(train_loader) + i)

                    print(
                        f"Worker {rank + 1}, Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], "
                        f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, "
                        f"Generator Loss: {gen_loss.item():.4f}")

                # Log images to TensorBoard every 100 steps
                if i % 100 == 0:
                    writer.add_images('Generated_Images', fake_images, epoch * len(train_loader) + i)

            # Step the profiler after each epoch to collect data

        dist.destroy_process_group()
        # # Generate and save images
        # def generate_and_save_images(model, epoch, noise):
        #     model.eval()
        #     with torch.no_grad():
        #         fake_images = model(noise).cpu()
        #         fake_images = fake_images.view(fake_images.size(0), 3, 64, 64)
        #         save_path = f'generated_images_epoch_{epoch}.png'
        #         save_image(fake_images, save_path)  # Save the generated images
        #         print(f"Generated image saved to {save_path}")

        # Generate test noise
        test_noise = torch.randn(16, NOISE_DIM, device=device)
        writer.close()
        
    except Exception as e:
        import traceback
        print(f"Exception occurred in process {rank}:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import os
    import random
    import numpy as np
    torch.multiprocessing.set_start_method('spawn', force=True)
    processes = []
    num_processes = multiprocessing.cpu_count()
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Localhost (use IP for distributed multi-node training)
    os.environ["MASTER_PORT"] = "29500"  # A random port number in the range 1024–65535 (avoid clashes)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["PYTHONHASHSEED"] = "42"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.spawn(
        main,  # Target function
        args=(num_processes,),  # Arguments passed to `main`
        nprocs=num_processes,  # Total number of processes
        join=True,  # Wait for processes to finish,
    )

    print("Training finished")
    #main()
