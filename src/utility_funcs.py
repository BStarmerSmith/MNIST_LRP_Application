from PIL import ImageEnhance, ImageOps
from src.variables import *
import numpy as np
import os
import torch.nn
import matplotlib.pyplot as plt
import copy
from torchsummary import summary


# This function takes in an PIL Image and converts it to a grayscale image to match MNIST format
# it then enhances the contrast of the image to make feature detection easier.
def format_image(image):
    image = image.convert(mode='L')
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


# This function simply takes an image and resizes it while making the previous content central.
# This works by getting the content that's not whitespace and padding around that.
def resize_and_center(sample, new_size=28):
    image = ImageOps.invert(sample)
    bbox = image.getbbox()
    crop = image.crop(bbox)
    delta_w = new_size - crop.size[0]
    delta_h = new_size - crop.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(crop, padding)


# This function is just used to remove some repetitive code that never changes.
def setup_lrp(image):
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.to(device)
    model.eval()
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    return model, image_tensor


# This function takes a list of Linear layers and converts them to Conv2D layers.
# The layer at index 0 needs to be specifically formatted to deal with the adjustment
# of a 2d network to a 1d network.
def toconv(layers):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 64, layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m, n, 7)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 7, 7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m, n, 1)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = torch.nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def newlayer(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer


def show_tensor(tensor, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.matshow(tensor)
    if title: plt.title(title)
    plt.show()


# This function is used to show an image of whatever tensor is presented.
def show_image_tensor(tensor, figsize=(8, 4), title=None):
    tensor = tensor.reshape(28, 28)
    plt.figure(figsize=figsize)
    plt.matshow(tensor, cmap='gray')
    if title: plt.title(title)
    plt.show()


# This function takes in an image, a figure size and a title and presents the image on a chart.
def show_image(image, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    if title: plt.title(title)
    plt.show()


# This function is used to print the model structure.
def print_model():
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.eval()
    model.to(device)
    print(summary(model, (1, 28, 28)))
    print(model)


# This function processes the data of the relevancy so its in the correct form
# to be presented.
def process_array(arr):
    output = []
    for lable, element in arr:
        output.append((lable, np.array(element.cpu()).sum(axis=0)))
    return output


# This function takes the 5 most likely outputs and presents them as a string for display.
def process_percentage(tuple):
    out_str = ""
    for index, percentage in tuple:
        if percentage == 1.0:
            out_str += "{}: 100% ".format(index)
        else:
            out_str += "{}: {:.5%} ".format(index, percentage)
    return out_str

