import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from src.variables import *
from PIL import ImageEnhance, ImageOps, Image
from torch.autograd import Variable
from torchvision import transforms

MINST_MEAN, MINST_STANDARD_DIV = 0.1307, 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MINST_MEAN,), (MINST_STANDARD_DIV,))])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This function takes in an PIL Image and formats it to minsk format, then returns the image
# which has been processed.
def process_image(raw_image, new_size=28):
    image = format_image(raw_image)
    image = resize_and_center(image, new_size)
    #predicted_img = predict_image(image)
    preform_lrp(image)


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


# This function takes in an PIL Image, converts it into a tensor then uses the trained cnn to get
# predict the digit in the image.
def predict_image(image):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    cnn_input = cnn_input.to(device)
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.to(device)  # Ensures model is on either CPU or GPU
    output = model(cnn_input)
    output = output.to(device)  # Does the same
    _, predicted = torch.max(output.data, 1)
    return predicted.sum().item()


def preform_lrp(image):
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    show_image_tensor(image_tensor)
    length = 5
    A = process_data(model=model, image_tensor=image_tensor)


# This function takes the input and works out the output at each level.
def process_data(model, image_tensor):
    layer1 = model.layer1(image_tensor).data
    layer2 = model.layer2(layer1).data
    get_heatmap(data=layer1, layer="layer1")
    get_heatmap(data=layer2, layer="layer2")
    dropout = layer2.reshape(layer2.size(0), -1)
    dropout = model.drop_out(dropout).data
    fc1 = model.fc1(dropout).data
    fc2 = model.fc2(fc1).data
    return {"layer1":layer1, "layer2:":layer2,
            "dropout":dropout, "fc1":fc1, "fc2":fc2}


def get_heatmap(layer, data):
    print("-"*100)
    image_details, _ = torch.max(data, 1)
    print(image_details, _)
    image_details = image_details[0]
    print(image_details)
    show_tensor(image_details, title=layer)


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

