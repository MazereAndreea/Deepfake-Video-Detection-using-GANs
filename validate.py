import os
from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
import torchvision.datasets as dset
import torch.utils.data
import torchvision.utils as vutils

writer = SummaryWriter()


def validate(generator, discriminator, device, noise_dim, batch_size):
    generator.eval()
    discriminator.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    DATA_PATH = '/home/andreeamazere.am/data/validation'
    dataset = dset.ImageFolder(root=DATA_PATH, transform=transform)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    step = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            print(f"\n Processing Batch {batch_idx + 1}/{len(val_dataloader)}")

            real_images, _ = data
            real_images = real_images.to(device)
            batch = real_images.size(0)

            print(f" Real Images Shape: {real_images.shape}")

            # Validation on real images
            real_labels = torch.ones(batch, device=device) #creare 128 de labels cu 1
            real_outputs = discriminator(real_images).view(-1) #creare scor + in tensor 1d
            real_loss = criterion(real_outputs, real_labels) #creare gradiente
            real_preds = (real_outputs > 0.5).float() #clasifica in 1 pt real, 0 fals (format float)
            correct_real = (real_preds == 1).sum().item() #pt predictiile de 1 afisare

            print(f" Real Loss: {real_loss.item()} | Correct Real: {correct_real}")

            # Validation on fake images generated by the generator
            noise = torch.randn(batch, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)

            print(f" Fake Images Shape: {fake_images.shape}")

            fake_labels = torch.zeros(batch, device=device)
            fake_outputs = discriminator(fake_images).view(-1)
            fake_loss = criterion(fake_outputs, fake_labels)
            fake_preds = (fake_outputs <= 0.5).float()
            correct_fake = (fake_preds == 0).sum().item()

            print(f" Fake Loss: {fake_loss.item()} | Correct Fake: {correct_fake}")

            total_loss = real_loss + fake_loss
            val_loss += total_loss.item()
            correct += (correct_real + correct_fake)
            total += (batch * 2)

            print(f" Batch Total Loss: {total_loss.item()} | Running Accuracy: {correct / total:.4f}")

            # Log generated images for qualitative evaluation
            grid_img = vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True)
            writer.add_images("Generated Images", grid_img.unsqueeze(0))  # Add batch dimension
            # Log scalar metrics
            writer.add_scalar("Validation/Loss", correct_fake, step)
            writer.add_scalar("Validation/Accuracy", correct_real, step)
            step += 1

    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = correct / total if total > 0 else 0
    print(f"Validation Loss: {avg_val_loss} | Accuracy: {accuracy}")

    writer.close()
    torch.save(generator.state_dict(), 'generator_valid.pth')
    torch.save(discriminator.state_dict(), "discriminator_valid.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator().to(device)
generator = Generator().to(device)
discriminator.load_state_dict(torch.load('discriminator_30epochs.pth', map_location=device, weights_only=True))
generator.load_state_dict(torch.load('generator_30epochs.pth', map_location=device, weights_only=True))
validate(generator, discriminator, device, 100, 128)
