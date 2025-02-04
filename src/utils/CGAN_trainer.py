"""
Trainer class used to manage the training process of the model
"""

import datetime
import torch
import random
import numpy as np

from collections import defaultdict

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.models.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# --------------------------------------------------------------------------------
class Trainer:
    """ Trainer manages the training process and anything related to it (e.g. loading/saving weights)
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 data_root,
                 load_subset=None,
                 output_path=None,
                 batch_size=32,
                 lr=2e-4,
                 device=None,
                 wandb_watch=False,
                 **kwargs):
        """ Initialise the Trainer class

        ---
        Parameters
            data_root: Path to dataset
            load_subset: Load only subset of data, either num of samples (int) or proportion (float 0-1)
            output_path: Path where outputs will be saved
            batch_size: Data loader batch size
            lr: Learning rate
            device: Device, if None then CPU will be used
            wandb_watch: Watch model with wandb

        """

        # Output Path
        self.output_path = self.set_output_path(output_path=output_path)

        # Set config
        self.config = kwargs
        self.config["load_subset"] = load_subset
        self.config["output_path"] = self.output_path
        self.config["batch_size"] = batch_size
        self.config["lr"] = lr
        self.config["wandb_watch"] = wandb_watch

        # On CPU by  default
        self.device = torch.device("cpu") if device is None else device

        # Init Model, Dataset and Data Loader
        # Probably bit of an overkill but might be useful to have access to those in notebooks
        self.model = CycleGAN(lr=lr)
        self.dataset = MonetDataset(root_path=data_root,
                                    load_subset=load_subset)
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=4)

    # --------------------------------------------------------------------------------
    def fit_wandb(self,
                  **kwargs):
        """ Fit and initialise with wandb

        This will set wandb.run to not be None and log bunch of things

        ---
        Parameters
            **kwargs: same paramters as to self.fit()

        """

        with wandb.init(project="Monet_CycleGAN", config=self.config):
            self.fit(**kwargs)

    # --------------------------------------------------------------------------------
    def fit(self,
            num_epochs=1,
            save_weights=True,
            load_weights_path=None):
        """ Fit the model on dataset

        ---
        Parameters
            num_epochs: Number of trianing epochs
            save_weights: Whether to save weights after training is finished
            load_weights_path: Load weights from given path

        """

        # --- Model Setup
        self.model = self.model.to(self.device)
        self.model.setup()
        if load_weights_path is not None:
            self.load_weights(load_path=load_weights_path)

        # --- WandB settings
        if wandb.run is not None:
            wandb.config["num_epochs"] = num_epochs
            wandb.config["save_weights"] = save_weights
            wandb.config["weights_path"] = load_weights_path

            a = wandb.config
            if wandb.config.wandb_watch:
                wandb.watch(self.model,
                            log="all", log_freq=10)

        # --- Train
        trainig_iter = 0
        for i in range(num_epochs):
            # Tqdm
            with tqdm(total=len(self.data_loader),
                      desc=f"Epoch {i + 1}/{num_epochs}",
                      ascii=True,
                      colour="green",
                      dynamic_ncols=True) as pbar:

                # Per batch
                loss_batch = defaultdict(list)
                for x in self.data_loader:
                    x_monet = x[0].to(self.device)
                    x_photo = x[1].to(self.device)

                    # Batch of data
                    loss = self.model.forward_step((x_monet, x_photo))

                    # Update dict
                    for k, v in loss.items():
                        loss_batch[k].append(v.item())

                    # Bar Update
                    pbar.set_postfix({k: v.item() for k, v in loss.items()})
                    pbar.update()

                    # Step WandB Update
                    if wandb.run is not None:
                        loss = {f"step_{k}" : v for k, v in loss.items()}
                        wandb.log(loss, step=trainig_iter)

                    # Update iteration
                    trainig_iter += 1

                # Epoch WandB Update
                if wandb.run is not None:
                    mean_loss_batch = {f"epoch_{k}" : np.mean(v) for k, v in loss_batch.items()}
                    mean_loss_batch["epoch"] = i
                    wandb.log(mean_loss_batch, step=trainig_iter)

            # Save weights of the final model
            if save_weights:
                self.save_weights()

    # --------------------------------------------------------------------------------
    def save_weights(self,
                     save_path=None):
        """ Save model weights

        ---
        Parameters
            save_path: Save path for weights
        """

        # Default path is output_path
        if save_path is None:
            save_path = self.output_path

        save_path = Path(save_path) / "model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Weights saved at: {save_path}")

    # --------------------------------------------------------------------------------
    def load_weights(self,
                     load_path):
        """ Load model weights

        ---
        Parameters
            load_path: Load path for weights

        """

        self.model.load_state_dict(torch.load(load_path))
        print(f"Weights loaded from: {load_path}")

    # --------------------------------------------------------------------------------
    def set_output_path(self,
                        output_path=None):
        """ Make a default output path

        ---
        Parameters
            output_path: Output path

        """

        # Set output path if None given
        if output_path is None:
            # Format
            date_time = datetime.datetime.now()
            date_str = date_time.strftime('%Y_%m_%d')
            time_str = date_time.strftime('%H_%M_%S')

            # Set
            output_path = Path("output") / date_str / time_str
        else:
            output_path = Path(output_path)

        # Create if doesn't exist
        if not output_path.exists():
            output_path.mkdir(parents=True)

        return output_path