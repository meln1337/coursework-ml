import torch
import torch.nn as nn
from torch.utils.tensorboard  import SummaryWriter
from torchvision.utils import save_image

import os
import time

from src import config
from src.utils.NST_utils import save_video, inverse_normalize

def train_model(
    model: nn.Module,
    optimizer,
    device,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    generated_img: torch.Tensor,
    epochs: int,
    alpha: float,
    beta: float,
    print_every: int,
    writer: SummaryWriter,
    directory: str,
    fps: int
):
    history = {
        'loss': [],
        'time elapsed': 0
    }

    start_training = time.time()

    try:
        os.mkdir(directory)
    except OSError as err:
        print('The model has already been trained on these 2 images')

    image_folder = os.path.join(config.ROOT_DIR, directory)

    content_img = content_img.to(device)
    style_img = style_img.to(device)
    generated_img = generated_img.to(device)

    if writer is not None:
        writer.add_image('content image', content_img[0])
        writer.add_image('style image', style_img[0])

    for epoch in range(epochs):
        content_img_features = model(content_img)
        style_features = model(style_img)
        generated_features = model(generated_img)

        style_loss = content_loss = 0

        # iterate through all the features for the chosen layers
        for content_feature, style_feature, gen_feature in zip(
            content_img_features, style_features, generated_features
        ):
            # batch_size equals 1
            batch_size, channel, height, width = gen_feature.shape
            content_loss += torch.mean((gen_feature - content_feature) ** 2)
            # Computing Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            # Computing Gram Matrix of Style
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history['loss'].append(total_loss.item())

        if writer is not None:
            writer.add_scalar("Total loss", history['loss'][epoch], epoch)
            writer.add_image('generated image', generated_img[0], epoch)

        if config.NORMALIZE:
            save_image(inverse_normalize(generated_img), os.path.join(image_folder, f'{epoch + 1}.png'))
        else:
            save_image(generated_img, os.path.join(image_folder, f'{epoch+1}.png'))

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Loss = {total_loss.item()}')

    save_video(image_folder, fps)

    if writer is not None:
        writer.flush()

    end_training = time.time()

    history['time elapsed'] = end_training - start_training
    return history