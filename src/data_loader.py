import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from src.variables import *
from PIL import ImageEnhance, ImageOps, Image
import copy
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
    model.eval()
    model.to(device)  # Ensures model is on either CPU or GPU
    output = model(cnn_input)
    output = output.to(device)  # Does the same
    _, predicted = torch.max(output.data, 1)
    return predicted.sum().item()


def preform_lrp(image):
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.to(device)
    model.eval()
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    show_image_tensor(image_tensor)
    print(e_lrp(model, image_tensor))


def e_lrp(model, img_tensor):
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)
              and not isinstance(module, torch.nn.Dropout)][1:]  # Gets all layers except sequential and dropout
    L = len(layers)
    A = [img_tensor] + [img_tensor] * L  # Makes the list
    for layer in range(L):
        if layer == 6:  # This is where we need to reshape the tensor
            print(A[layer].shape)
            A[layer] = A[layer].reshape(A[layer].size(0), -1)
            print(A[layer].shape)
        A[layer + 1] = layers[layer].forward(A[layer].to(device))

    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))  # The index of the highest classification
    T = torch.FloatTensor(np.abs(np.array(T)) * 0)  # Sets all values to 0 - as a tensor
    T[index] = 1  # Sets the index of the highest classification to 1, as its all we care for

    R = [None] * L + [(A[-1].cpu() * T).data + 1e-6]
    for layer in range(0, L)[::-1]:
        if isinstance(layers[layer], torch.nn.Conv2d) or isinstance(layers[layer], torch.nn.Linear):
            # Specifies the rho function that will be applied to the weights of the layer
            if 0 < layer <= 5:  # Gamma rule (LRP-gamma)
                rho = lambda p: p + 0.25 * p.clamp(min=0)
            else:  # Basic rule (LRP-0)
                rho = lambda p: p
            print(layer)
            A[layer] = A[layer].data.requires_grad_(True)
            # Step 1: Transform the weights of the layer and executes a forward pass
            z = newlayer(layers[layer], rho).forward(A[layer]) + 1e-9
            # Step 2: Element-wise division between the relevance of the next layer and the denominator
            s = (R[layer + 1].to(device) / z).data
            # Step 3: Calculate the gradient and multiply it by the activation layer
            (z * s).sum().backward()
            c = A[layer].grad
            R[layer] = (A[layer] * c).cpu().data
            if layer == 6:  # Reshape after layer fc1
                R[layer] = torch.reshape(R[layer], (1, 64, 7, 7))
                print(R[layer].shape)
        else:
            R[layer] = R[layer + 1]

    # Return the relevance of the input layer
    return R[0]

def newlayer(layer, rho):
    layer = copy.deepcopy(layer)
    layer.weight = torch.nn.Parameter(rho(layer.weight))
    layer.bias = torch.nn.Parameter(rho(layer.bias))
    return layer

def get_heatmap(data, layer=""):
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

