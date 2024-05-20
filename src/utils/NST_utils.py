import torch
import torchvision.transforms as T
from src import config
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
from moviepy.video.io import ImageSequenceClip
import cv2

if config.NORMALIZE:
    transform = T.Compose([
            T.Resize((config.HEIGHT, config.WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
else:
    transform = T.Compose([
        T.Resize((config.HEIGHT, config.WIDTH)),
        T.ToTensor()
    ])

inverse_normalize = T.Normalize(
    mean=config.INVERSE_IMAGENET_MEAN,
    std=config.INVERSE_IMAGENET_STD
)

def load_img(path):
    img = Image.open(path)
    # img = transform(img)
    return img

def plot_2_img(content_img, style_img):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(content_img)
    axes[0].set_title('Content image')
    axes[1].imshow(style_img)
    axes[1].set_title('Style image')

    path_dir = os.path.join(config.ROOT_DIR, config.OUTPUT_NST_DIR, concat_images(config.MAIN_CONTENT_IMG_EXAMPLE, config.MAIN_STYLE_IMG_EXAMPLE))
    path_file = os.path.join(path_dir, 'content and style images.png')

    try:
        os.mkdir(path_dir)
    except OSError as err:
        print('The model has already been trained on these 2 images')

    plt.savefig(path_file)
    plt.show()

def plot_3_img(content_img, style_img, generated_img):
    if isinstance(content_img, torch.Tensor):
        content_img = content_img.detach().cpu().permute(1, 2, 0)
    if isinstance(style_img, torch.Tensor):
        style_img = style_img.detach().cpu().permute(1, 2, 0)
    if isinstance(generated_img, torch.Tensor):
        generated_img = generated_img.detach().cpu().permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(content_img)
    axes[0].set_title('Content image')
    axes[1].imshow(style_img)
    axes[1].set_title('Style image')
    axes[2].imshow(generated_img)
    axes[2].set_title('Generated image')

    fig.tight_layout()

    path_dir = os.path.join(config.ROOT_DIR, config.OUTPUT_NST_DIR,
                            concat_images(config.MAIN_CONTENT_IMG_EXAMPLE, config.MAIN_STYLE_IMG_EXAMPLE))
    path_file = os.path.join(path_dir, 'content, style and gen images.png')

    try:
        os.mkdir(path_dir)
    except OSError as err:
        print('The model has already been trained on these 2 images')

    plt.savefig(path_file)

    plt.show()

def plot_loss(history: dict):
    loss = history['loss']
    plt.plot(loss)
    plt.title('Loss during the training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    path_dir = os.path.join(config.ROOT_DIR, config.OUTPUT_NST_DIR,
                            concat_images(config.MAIN_CONTENT_IMG_EXAMPLE, config.MAIN_STYLE_IMG_EXAMPLE))
    path_file = os.path.join(path_dir, 'loss.png')

    try:
        os.mkdir(path_dir)
    except OSError as err:
        print('The model has already been trained on these 2 images')

    plt.savefig(path_file)

    plt.show()

def concat_images(content_img, style_img):
    return f"{content_img.split('.')[0]} {style_img.split('.')[0]}"

def save_video(dir_path: str, fps: int):
    for epoch in range(config.EPOCHS):
        file = f'{epoch+1}.png'
        img = cv2.imread(os.path.join(dir_path, file))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (85, 220)
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        img = cv2.putText(img, f'Epoch = {epoch+1}', org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(os.path.join(dir_path, file), img)

    image_files = [os.path.join(dir_path, f'{epoch+1}.png')
                   for epoch in range(config.EPOCHS)]
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(dir_path, 'timelapse.mp4'))