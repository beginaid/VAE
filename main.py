import os

import fire
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from libs.Visualize import Visualize
from models.VAE import VAE


class Main():
    def __init__(self, z_dim):
        """Constructor

        Args:
            z_dim (int): Dimensions of the latent variable.

        Returns:
            None.
        """
        self.z_dim = z_dim
        self.dataloader_train = None
        self.dataloader_valid = None
        self.dataloader_test = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(self.z_dim).to(self.device)
        self.writer = SummaryWriter(log_dir="./logs")
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_max_epochs = 3
        self.num_no_improved = 0
        self.num_batch_train = 0
        self.num_batch_valid = 0
        self.loss_valid_min = 10 ** 7 # Initialize with a large value
        self.Visualize = Visualize(self.z_dim, self.dataloader_test, self.model, self.device)


    def createDirectories(self):
        """Create directories for the tensorboard and learned model

        Args:
            None.

        Returns:
            None.
        """
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        if not os.path.exists("./params"):
            os.makedirs("./params")

    def createDataLoader(self):
        """Download MNIST and convert it to data loaders

        Args:
            None.

        Returns:
            None.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]) # Preprocessing for MNIST images
        dataset_train_valid = datasets.MNIST("./", train=True, download=True, transform=transform) # Separate train data and test data to get a dataset
        dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

        # Use 20% of train data as validation data
        size_train_valid = len(dataset_train_valid) # 60000
        size_train = int(size_train_valid * 0.8) # 48000
        size_valid = size_train_valid - size_train # 12000
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

        # Create dataloaders from the datasets
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=True)
        self.dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1000, shuffle=False)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False)
        self.Visualize.dataloader_test = self.dataloader_test

    def train_batch(self):
        """Batch-based learning for training data

        Args:
            None.

        Returns:
            None.
        """
        self.model.train()
        for x, _ in self.dataloader_train:
            lower_bound, _, _ = self.model(x, self.device)
            loss = -sum(lower_bound)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("Loss_train/KL", -lower_bound[0].cpu().detach().numpy(), self.num_iter + self.num_batch_train)
            self.writer.add_scalar("Loss_train/Reconst", -lower_bound[1].cpu().detach().numpy(), self.num_iter + self.num_batch_train)
            self.num_batch_train += 1
        self.num_batch_train -= 1

    def valid_batch(self):
        """Batch-based learning for validating data

        Args:
            None.

        Returns:
            None.
        """
        loss = []
        self.model.eval()
        for x, _ in self.dataloader_valid:
            lower_bound, _, _ = self.model(x, self.device)
            loss.append(-sum(lower_bound).cpu().detach().numpy())
            self.writer.add_scalar("Loss_valid/KL", -lower_bound[0].cpu().detach().numpy(), self.num_iter + self.num_batch_valid)
            self.writer.add_scalar("Loss_valid/Reconst", -lower_bound[1].cpu().detach().numpy(), self.num_iter + self.num_batch_valid)
            self.num_batch_valid += 1    
        self.num_batch_valid -= 1
        self.loss_valid = np.mean(loss)
        self.loss_valid_min = np.minimum(self.loss_valid_min, self.loss_valid)

    def early_stopping(self):
        """Judging early stopping

        Args:
            None.

        Returns:
            None.
        """
        if self.loss_valid_min < self.loss_valid: # If the loss of this iteration is greater than the minimum loss of the previous iterations, the counter variable is incremented.
            self.num_no_improved += 1
            print(f"Validation got worse for the {self.num_no_improved} time in a row.")
        else: # If the loss of this iteration is the same or smaller than the minimum loss of the previous iterations, reset the counter variable and save parameters.
            self.num_no_improved = 0
            torch.save(self.model.state_dict(), f"./params/model_z_{self.z_dim}.pth")

    def main(self):
        self.createDirectories()
        self.createDataLoader()
        print("-----Start training-----")
        for self.num_iter in range(self.num_max_epochs):
            self.train_batch()
            self.valid_batch()
            print(f"[EPOCH{self.num_iter + 1}] loss_valid: {int(self.loss_valid)} | Loss_valid_min: {int(self.loss_valid_min)}")
            self.early_stopping()
            if self.num_no_improved >= 10:
                print("Apply early stopping")
                break
        self.writer.close()
        print("-----Stop training-----")
        print("-----Start Visualization-----")
        self.model.load_state_dict(torch.load(f"./params/model_z_{self.z_dim}.pth"))
        self.model.eval()
        self.Visualize.createDirectories()
        self.Visualize.reconstruction()
        self.Visualize.latent_space()
        self.Visualize.lattice_point()
        self.Visualize.walkthrough()
        print("-----Stop Visualization-----")
    
if __name__ == '__main__':
    fire.Fire(Main)
