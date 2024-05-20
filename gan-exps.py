import torch
from src.models.cycle_gan import CycleGAN
from src.dataset.monet_dataset import MonetDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from_tensor = lambda x: ((x.squeeze().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8)
from_tensors = lambda x: ((x.squeeze().permute(0, 2,3,1).numpy() + 1.0) * 127.5).astype(np.uint8)
read_img = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 127.5 - 1.0

dataset_path = 'C:/Projects/jupyter projects/ml course/coursework/ISPM_dataset'
model_path = 'C:/Projects/jupyter projects/ml course/coursework\model.pth'

NUM_STEPS = 1

BATCH_SIZE = 16

dataset = MonetDataset(root_path=dataset_path)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = CycleGAN().to(device)
try:
    model.load_state_dict(torch.load(model_path), strict=False)
    print('The model has been loaded')
except:
    print('Could not load the model')

model.eval()

it = iter(dataloader)

for i in range(NUM_STEPS):
    x = next(it)

    x_monet = x[0].to(device)
    x_photo = x[1].to(device)
    # with torch.inference_mode(mode=True):
    #     monet_pred = model(x_monet)
    monet_pred = model(x_monet)

rows = 4
cols = 4


fig, axes = plt.subplots(rows, cols, figsize=(12, 9))

k = 0

monet_pred = from_tensors(monet_pred.detach().cpu())
print(monet_pred)
print(monet_pred.shape)

for row in range(rows):
    for col in range(cols):
        axes[row, col].imshow(monet_pred[k])
        k += 1

fig.tight_layout()
plt.show()