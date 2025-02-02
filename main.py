import os
from discriminator import Discriminator
from generator import Generator
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torch.utils.data
import torchvision.utils as vutils


def sendData(smth=None):
    return smth


NOISE_DIM = 100
NUM_EPOCHS = 30
BATCH_SIZE = 128


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    writer = SummaryWriter(log_dir="runs/generative_model")
    generator = Generator()
    discriminator = Discriminator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    DATA_PATH = '/home/andreeamazere.am/data/celeba'
    dataset = dset.ImageFolder(root=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()

    iters = 0
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            real_images, _ = data
            real_images = real_images.to(device)

            discriminator.zero_grad()
            labels = torch.full((real_images.size(0),), np.random.uniform(0.8, 1.0), dtype=torch.float, device=device)
            real_outputs = discriminator(real_images)
            real_outputs = real_outputs.view(-1)
            real_loss = criterion(real_outputs, labels)
            real_loss.backward()
            D_real = real_outputs.mean().item()

            noise = torch.randn(real_images.size(0), NOISE_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            labels = torch.full((real_images.size(0),), np.random.uniform(0.0, 0.2), dtype=torch.float, device=device)
            fake_outputs = discriminator(fake_images.detach())
            fake_outputs = fake_outputs.view(-1)
            fake_loss = criterion(fake_outputs, labels)
            fake_loss.backward()
            D_fake_before_update_generator = fake_outputs.mean().item()
            discriminator_loss = real_loss + fake_loss
            discriminator_optimizer.step()

            generator.zero_grad()
            labels.fill_(1)
            gen_outputs = discriminator(fake_images)
            gen_outputs = gen_outputs.view(-1)
            gen_loss = criterion(gen_outputs, labels)
            gen_loss.backward()
            fake_images_after_update = generator(noise)
            D_fake_after_update_generator = discriminator(fake_images_after_update).mean().item()
            generator_optimizer.step()

            if i % 50 == 0:
                writer.add_scalar("Loss/Discriminator", discriminator_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar("Loss/Generator", gen_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar("Probability/D_real", D_real, epoch * len(dataloader) + i)
                writer.add_scalar("Probability/D_fake_before_update_generator", D_fake_before_update_generator,
                                  epoch * len(dataloader) + i)
                writer.add_scalar("Probability/D_fake_after_update_generator", D_fake_after_update_generator,
                                  epoch * len(dataloader) + i)
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(dataloader)}], '
                      f'Discriminator Loss: {discriminator_loss.item():.4f}, '
                      f'Generator Loss: {gen_loss.item():.4f}')

            if (iters % 500 == 0):
                with torch.no_grad():
                    noise = torch.randn(64, NOISE_DIM, 1, 1, device=device)
                    fake_images = generator(noise).detach().cpu()

                    # Create a grid of images
                    grid_img = vutils.make_grid(fake_images, padding=2, normalize=True)

                    grid_img = grid_img.unsqueeze(0)
                    # Log the images to TensorBoard
                    writer.add_images(f"Generated Images/img", grid_img, epoch)
    iters += 1

    # # Generate and save images
    # def generate_and_save_images(writer, model):
    #     model.eval()
    #     with torch.no_grad():
    #         noise = torch.randn(64, NOISE_DIM, 1, 1, device=device)
    #         fake_images = model(noise).detach().cpu()
    #
    #         # Create a grid of images
    #         grid_img = vutils.make_grid(fake_images, padding=2, normalize=True)
    #
    #         grid_img = grid_img.unsqueeze(0)
    #         # Log the images to TensorBoard
    #         writer.add_images(f"Generated Images/img", grid_img, epoch)
    #
    # generate_and_save_images(writer, generator)
    writer.close()
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict()
    }, "model_checkpoint.pth")
    torch.save(discriminator.state_dict(), "discriminator_train.pth")

if __name__ == "__main__":
    main()