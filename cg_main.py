"""
Main entrypoint, for now just training
"""
import argparse
import torch

from src.utils.CGAN_trainer import Trainer

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- Args
    parser = argparse.ArgumentParser(description='Simple implementation of CycleGAN in PyTorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', type=str, default="C:/Projects/jupyter projects/ml course/coursework/ISPM_dataset",
                        help='Root path to dataset')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,  # Max that fits onto 24GB GPU
                        help='Batch size for processing data, mine\'s set to max for 3090 24GB')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weights_path', type=str, default="C:/Projects/jupyter projects/ml course/coursework/output/CGAN",  # Path to load weights
                        help='Weights path, loads only if not None')
    parser.add_argument('--output_path', type=str, default="C:/Projects/jupyter projects/ml course/coursework/output/CGAN",  # Path to saving data e.g. weights
                        help='Save all data in "." if None')

    args = parser.parse_args()

    # Init trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    trainer = Trainer(device=device,
                      **vars(args))