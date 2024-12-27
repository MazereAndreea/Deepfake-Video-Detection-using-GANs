import os
from discriminator import Discriminator
from generator import Generator
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from HDF5Dataset import HDF5Dataset, preprocess_and_save_to_hdf5
from torch.utils.tensorboard import SummaryWriter

def sendData(smth=None):
    return smth

NOISE_DIM = 100
NUM_EPOCHS = 5
BATCH_SIZE = 128
HDF5_FILE = "preprocessed_dataset.h5"

def main():
    writer = SummaryWriter(log_dir="runs/generative_model")
    generator = Generator(NOISE_DIM)
    discriminator = Discriminator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    sendData(generator)

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # CelebA dataset
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Check if the HDF5 file exists
    if not os.path.exists(HDF5_FILE):
        # Preprocess dataset and save if not already cached
        train_dataset = torchvision.datasets.CelebA(
            root='C:/Users/ANDREEA/PycharmProjects/Rn/data/',
            split='train',
            transform=transform,
            download=False
        )
        preprocess_and_save_to_hdf5(train_dataset, HDF5_FILE)

    # Load data from HDF5
    train_dataset = HDF5Dataset(HDF5_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
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
                writer.add_scalar("Loss/Discriminator", real_loss.item() + fake_loss.item(),epoch * len(train_loader) + i)
                writer.add_scalar("Loss/Generator", gen_loss.item(), epoch * len(train_loader) + i)
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], '
                      f'Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, '
                      f'Generator Loss: {gen_loss.item():.4f}')

    # Generate and save images
    def generate_and_save_images(writer, model, epoch):
        model.eval()
        with torch.no_grad():
            noise = torch.randn(16, NOISE_DIM, device=device)
            fake_images = model(noise).cpu()

            # Normalize the images to [0, 1] for TensorBoard
            fake_images = (fake_images + 1) / 2

            # Log the images to TensorBoard
            writer.add_images(f"Generated Images/Epoch {epoch + 1}", fake_images, epoch)

    generate_and_save_images(writer, generator, epoch)
    writer.close()

if __name__ == "__main__":
    main()