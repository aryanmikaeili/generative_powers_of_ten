import PIL.Image
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def visualize_images(images, row_size):
    #images: list of PIL images
    #row_size: number of images per row
    #returns: PIL image

    num_images = len(images)
    col_size = num_images // row_size
    if num_images % row_size != 0:
        col_size += 1

    image_size = images[0].size[0]
    canvas = Image.new('RGB', (image_size * row_size, image_size * col_size))
    for i, image in enumerate(images):
        canvas.paste(image, (image_size * (i % row_size), image_size * (i // row_size)))
    return canvas

def crop_center(image):
    h, w = image.shape[:2]
    crop_size = min(h, w)

    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return Image.fromarray(image[start_h:start_h + crop_size, start_w:start_w + crop_size])

def load_image(image_paths, image_size = 512, to_tensor = False, is_mask = False, device = 'cuda', dtype = torch.float32, process_mask = True):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    res = []
    res_tensor = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert('L' if is_mask else 'RGB'))
        image = crop_center(image)
        image = image.resize((image_size, image_size), Image.BILINEAR)
        if is_mask:
            if process_mask:
                image = Image.fromarray(np.array(image) < (0.5 * 255)).convert('L')
            else:
                image = Image.fromarray(np.array(image) > (0.5 * 255)).convert('L')
        res.append(image)

        if to_tensor:
            if not is_mask:
                image_tensor = TF.to_tensor(image) * 2 - 1.
            else:
                image_tensor = TF.to_tensor(image)
            image_tensor = image_tensor.to(device, dtype)
            res_tensor.append(image_tensor)

    return res, res_tensor


def rgba2rgb(image):
    #image: np.array of shape (H, W, 4) in range [0, 255]
    #returns: np.array of shape (H, W, 3) in range [0, 255]
    assert image.ndim == 3 and image.shape[2] == 4
    alpha = image[:, :, 3] / 255.
    rgb = image[:, :, :3]
    image = (1 - alpha[:, :, None]) * 255. + alpha[:, :, None] * rgb
    return image.astype(np.uint8)

def visualize_ca_maps(image, ca_maps):
    #image: PIL image
    #ca_maps: torch tensor of shape(B, H, W)
    #returns pil list

    image = np.array(image)
    ca_list = list(ca_maps)
    res = []
    for i in range(len(ca_list)):
        plt.figure(figsize=(10, 10))
        #turn off axis
        plt.axis('off')
        plt.imshow(image)
        curr_ca = ca_list[i].float().cpu().detach().numpy()
        plt.imshow(curr_ca, alpha = 0.5, cmap='jet')
        canvas = plt.gcf().canvas
        canvas.draw()
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        res.append(pil_image.resize((512, 512), Image.BILINEAR))
        plt.close()

    return res

def tensor2pil_list(images):
    #images: torch tensor of shape (B, 3, H, W) in range(-1, 1)

    res = []
    for i in range(images.shape[0]):
        current = images[i].permute(1, 2, 0).cpu().detach().numpy()
        current = (current + 1) / 2 * 255
        current = current.astype(np.uint8)
        res.append(Image.fromarray(current))

    return res

def merge_masks(masks):
    #masks: torch tensor of size (M, H, W). mask values are (0, 1)
    #returns: torch tensor of size (H, W)

    res = torch.zeros_like(masks[0])
    for i in range(len(masks)):
        res[masks[i] == 1] = 1

    return res