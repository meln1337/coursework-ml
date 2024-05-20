import torch.optim as optim

import os

from src.utils.NST_utils import load_img, transform, plot_2_img, concat_images, plot_3_img, plot_loss, inverse_normalize
from src.utils.NST_trainer import train_model
from src import config
from src.models.VGG import VGG

# writer to log info
# writer = SummaryWriter()

# Loading the images
content_img = load_img(os.path.join(config.ROOT_DIR, config.CONTENT_DIR, config.MAIN_CONTENT_IMG_EXAMPLE))
style_img = load_img(os.path.join(config.ROOT_DIR, config.STYLE_DIR, config.MAIN_STYLE_IMG_EXAMPLE))
# Ploting the images
plot_2_img(content_img, style_img)
# Defining the model
model = VGG().to(config.DEVICE).eval()


content_img = transform(content_img).unsqueeze(0).to(config.DEVICE)
style_img = transform(style_img).unsqueeze(0).to(config.DEVICE)
generated_img = content_img.clone().requires_grad_(True)

# content_img = content_img
# style_img = style_img.to(config.DEVICE)
# generated_img = generated_img.to(config.DEVICE)

optimizer = optim.Adam([generated_img], lr=config.LR)

directory = os.path.join(config.ROOT_DIR,
                         config.OUTPUT_NST_DIR,
                         concat_images(config.MAIN_CONTENT_IMG_EXAMPLE, config.MAIN_STYLE_IMG_EXAMPLE))

history = train_model(model, optimizer, config.DEVICE,
                      content_img, style_img, generated_img,
                      config.EPOCHS, config.ALPHA, config.BETA, config.PRINT_EVERY,
                      None, directory, config.FPS)

print(f'Time of training = {history["time elapsed"]} seconds')

# Ploting the images


if config.NORMALIZE:
    plot_3_img(inverse_normalize(content_img[0]), inverse_normalize(style_img[0]), inverse_normalize(generated_img[0]))
else:
    plot_3_img(content_img[0], style_img[0], generated_img[0])
# plot loss
plot_loss(history)

# writer.close()