import PIL
import cv2
import torchvision.transforms
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
import torch
import torchvision.transforms.functional as TF

from utils import visualize_images, tensor2pil_list



def downsample(image, p):
    # image: torch tensor of size (C, H, W)
    # p: downsampling factor

    image_c = image.clone()
    # apply gaussian blur
    image_c = TF.gaussian_blur(image_c, kernel_size=(p // 2) * 2 + 1, sigma=p / 2)
    # downsample
    image_c = image_c[:, ::p, ::p]

    return image_c


def padded_downsample(image, p):
    #image: torch tensor of size (C, H, W)
    #p: downsampling factor
    #downsamples image and pads it to the original size plus a mask that is one in the downsampled region and zero in the padded region

    C, H, W = image.shape
    #image = TF.resize(image, (H // p, W // p), interpolation = torchvision.transforms.InterpolationMode.BILINEAR , antialias=False)
    #image = downsample(image, p)
    #apply gaussian blur first
    image = TF.gaussian_blur(image, kernel_size=(p // 2) * 2 + 1)
    #avg_pool = torch.nn.AdaptiveAvgPool2d((H // p, W // p))
    image = image[:, ::p, ::p]
    #image = avg_pool(image.unsqueeze(0)).squeeze(0)
    #image = F.interpolate(image.unsqueeze(0), scale_factor=1 / p, mode='bilinear', antialias=True).squeeze(0)
    image = TF.pad(image, (W // 2 - W // (2 * p), H // 2 - H // (2 * p), W - W // 2 - W // (2 * p), H - H // 2 - H // (2 * p)))

    #create mask
    mask = torch.zeros_like(image)
    mask[:, H // 2 - H // (2 * p):H // 2 + H // (2 * p), W // 2 - W // (2 * p):W // 2 + W // (2 * p)] = 1

    #resize image and mask to H, W
    image = TF.resize(image, (H, W), interpolation = PIL.Image.BILINEAR)
    mask = TF.resize(mask, (H, W), interpolation = PIL.Image.BILINEAR)
    return image, mask
def paste_center(image_bg, image_fg, p, is_noise=False):
    # image_bg: tensor of size (C, H, W)
    # image_fg: tensor of size (C, H, W)
    # p: downsampling factor

    H, W = image_bg.shape[-2:]
    top_left = (H // 2 - H // (2 * p), W // 2 - W // (2 * p))
    image_fg_downsampled = downsample(image_fg, p)
    # image_fg_downsampled = TF.resize(image_fg, (H // p, W // p), interpolation = PIL.Image.BICUBIC, antialias=True)
    if is_noise:
        image_fg_downsampled *= p  # image_fg_downsampled.std() #p((p // 2) * 2 + 1)
    image_bg[:, top_left[0]:top_left[0] + image_fg_downsampled.shape[-2],
    top_left[1]:top_left[1] + image_fg_downsampled.shape[-1]] = image_fg_downsampled

    return image_bg




def zoom_in_image(image, p, resize_to_original=True):
    # image: PIL image or torch tensor of shape (C, H, W)
    # p: downsampling factor

    if isinstance(image, torch.Tensor):
        C, H, W = image.shape
    else:
        H, W = image.size
    top_left = (H // 2 - H // (2 * p), W // 2 - W // (2 * p))
    if resize_to_original:
        zoomed_image = TF.resized_crop(image, top_left[0], top_left[1], H // p, W // p, (H, W), antialias=True,
                                       interpolation=PIL.Image.BILINEAR)
    else:
        zoomed_image = TF.crop(image, top_left[0], top_left[1], H // p, W // p)

    return zoomed_image




def create_gaussian_pyramid(image, num_levels, p):
    # image:torch tensor of size (C, H, W)
    # num_levels: number of levels in pyramid
    # p: downsampling factor

    pyramid = [image]
    for i in range(1, num_levels):
        pyramid.append(downsample(pyramid[-1], p))

    return pyramid


def create_laplace_pyramid(image, num_levels, p, normalize=False):
    # image:torch tensor of size (C, H, W)
    # num_levels: number of levels in pyramid
    # p: downsampling factor

    gaussian_pyramid = create_gaussian_pyramid(image, num_levels, p)

    pyramid = []
    for i in range(num_levels - 1):
        upsampled = TF.resize(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape[-2:], interpolation=PIL.Image.BILINEAR)

        d = gaussian_pyramid[i] - upsampled
        if normalize:
            d = (d - d.min()) / (d.max() - d.min())

        pyramid.append(d)

    pyramid.append(gaussian_pyramid[-1])

    return pyramid


def reconstruct_image_from_laplacian_pyramid(pyramid, p):
    # pyramid: list of torch tensors
    # p: downsampling factor

    res = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        res = TF.resize(res, pyramid[i].shape[-2:], interpolation=PIL.Image.BILINEAR)
        res += pyramid[i]

    return res




def multi_resolution_blending(stack, p):
    # stack: torch tensor of size (L, C, H, W)
    # returns: torch tensor of size (L, C, H, W)
    # Blends frequency levels using laplacian pyramid blending

    L, C, H, W = stack.shape
    zoom_stack = []
    for i in range(L):
        pyramids = []
        for j in range(i + 1):
            current = stack[j]
            # crop center of current
            crop_level = p ** (i - j)
            top_left = (H // 2 - H // (2 * crop_level), W // 2 - W // (2 * crop_level))
            current = current[:, top_left[0]:top_left[0] + H // crop_level, top_left[1]:top_left[1] + W // crop_level]
            # upsample current
            # current = TF.resize(current, (H, W), interpolation = PIL.Image.BILINEAR)
            # build laplacian pyramid
            pyramid = create_laplace_pyramid(current, num_levels=j + 1, p=p, normalize=False)
            pyramids.append(pyramid)
        # average laplacians in the same frequancy level

        laplacian_pyramid = []
        num_levels = i + 1
        for n in range(num_levels):
            current_level = []
            for j in range(n, num_levels):
                current_level.append(pyramids[j].pop(-1))
            current_level = torch.stack(current_level)
            current_level = current_level.mean(dim=0)
            laplacian_pyramid.append(current_level)
        laplacian_pyramid.reverse()
        # reconstruct image
        res = reconstruct_image_from_laplacian_pyramid(laplacian_pyramid, p)

        zoom_stack.append(res)
        #stack[i] = res
    return torch.stack(zoom_stack)


def multi_resolution_blending2(stack, p):
    # stack: torch tensor of size (L, C, H, W)
    # returns: torch tensor of size (L, C, H, W)
    # Blends frequency levels using laplacian pyramid blending

    L, C, H, W = stack.shape
    zoom_stack = []
    for i in range(L):
        pyramids = []
        for j in range(i + 1):
            current = stack[j]
            # crop center of current
            crop_level = p ** (i - j)
            current = zoom_in_image(current, crop_level, resize_to_original=True)
            # upsample current
            # current = TF.resize(current, (H, W), interpolation = PIL.Image.BILINEAR)
            # build laplacian pyramid
            pyramid = create_laplace_pyramid(current, num_levels=L, p=p, normalize=False)
            pyramids.append(pyramid)

        # average laplacians in the same frequancy level
        pyramids.reverse()
        laplacian_pyramid = []
        num_levels = L
        pyramids.reverse()
        for n in range(num_levels):
            current_level = []
            for j in range(len(pyramids)):
                pyr = pyramids[j].pop(0)
                if j < n + 1:
                    current_level.append(pyr)
            current_level = torch.stack(current_level)
            current_level = current_level.mean(dim=0)
            laplacian_pyramid.append(current_level)
        # reconstruct image
        res = reconstruct_image_from_laplacian_pyramid(laplacian_pyramid, p)

        zoom_stack.append(res)

    return torch.stack(zoom_stack)

def multi_resolution_blending3(stack, p):
    # stack: torch tensor of size (L, C, H, W)
    # returns: torch tensor of size (L, C, H, W)
    # Blends frequency levels using laplacian pyramid blending

    L, C, H, W = stack.shape
    zoom_stack = []
    for i in range(L):
        pyramids = []
        for j in range(i + 1):
            current = stack[j]
            # crop center of current
            crop_level = p ** (i - j)
            current = zoom_in_image(current, crop_level, resize_to_original=True)
            # upsample current
            # current = TF.resize(current, (H, W), interpolation = PIL.Image.BILINEAR)
            # build laplacian pyramid
            pyramid = create_laplace_pyramid(current, num_levels=i + 1, p=p, normalize=False)
            pyramids.append(pyramid)
        # average laplacians in the same frequancy level

        pyramids.reverse()
        laplacian_pyramid = []
        num_levels = i + 1
        for n in range(num_levels):
            current_level = []
            for j in range(num_levels):
                pyr = pyramids[j].pop(0)
                if j < n + 1:
                    current_level.append(pyr)
            current_level = torch.stack(current_level)
            current_level = current_level.mean(dim=0)
            laplacian_pyramid.append(current_level)

        # reconstruct image
        res = reconstruct_image_from_laplacian_pyramid(laplacian_pyramid, p)

        zoom_stack.append(res)
        stack[i] = res

    return torch.stack(zoom_stack)
def render_zoom_stack(zoom_stack, level, p, is_noise=False):
    # zoom_stack: list of PIL images
    # level: level of zoom stack to render
    # p: downsampling factor

    res = zoom_stack[level]
    N = len(zoom_stack)
    l = p
    for i in range(level + 1, N):
        fg = zoom_stack[i]
        res = paste_center(res, fg, l, is_noise=is_noise)
        l *= p

    return res

def render_full_zoom_stack(zoom_stack, p, is_noise=False):
    # zoom_stack: tensor of size (L, C, H, W)
    # returns: consistent zoom stack of size (L, C, H, W)

    for i in range(len(zoom_stack)):
        zoom_stack[i] = render_zoom_stack(zoom_stack, i, p=p, is_noise=is_noise)

    return zoom_stack


if __name__ == '__main__':
    image = Image.open('data/man/image.png')
    image = TF.to_tensor(image) * 2 - 1
    image = image.to(device = 'cuda', dtype=torch.float32)


    # #
    N = 3
    zoom_stack = [image]
    for i in range(1, N):
        zoom_stack.append(zoom_in_image(zoom_stack[-1], 2))

    zoom_stack = torch.stack(zoom_stack)
    res = multi_resolution_blending3(zoom_stack, 2)

    visualize_images(tensor2pil_list(res), 2).save('multi_resolution_blending.png')
    # a = 0
    #
    #
    # N = 4
    # # zoom stack
    # zoom_stack = [image]
    # for i in range(1, N):
    #     zoom_stack.append(zoom_in_image(zoom_stack[-1], 2))
    #
    # stack = multi_resolution_blending(torch.stack(zoom_stack), 2)
    # stack = [TF.to_pil_image((stack[i] + 1) / 2 ) for i in range(stack.shape[0])]
    #
    # visualize_images(stack, 4).save('multi_resolution_blending.png')
    # a = 0

    # pyramid = create_laplace_pyramid(image, 4, 2, normalize = False )
    # res = reconstruct_image_from_laplacian_pyramid(pyramid, 2)
    # res = TF.to_pil_image((res + 1) * 0.5)
    # res.save('laplace_pyramid.png')
    #
    # pyramid = [TF.to_pil_image(image) for image in pyramid]
    #
    # visualize_images(pyramid, 4).save('laplace_pyramid.png')
    # N = 2
    # # zoom stack
    # zoom_stack = [image]
    # for i in range(1, N):
    #     zoom_stack.append(zoom_in_image(zoom_stack[-1], 2))
    #
    # #res = render_zoom_stack(zoom_stack, 0, 5)
    # zoom_stack = torch.stack(zoom_stack)
    # #res = multi_resolution_blending(zoom_stack, 2)
    # res = [TF.to_pil_image((zoom_stack[i] + 1) / 2) for i in range(zoom_stack.shape[0])]
    # visualize_images(res, 1).save('zoom_stack.png')
    # a = 0