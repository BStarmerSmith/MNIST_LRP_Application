import matplotlib.pyplot as plt
import torch
import os
from src.variables import *
from PIL import ImageEnhance, ImageOps
from torch.autograd import Variable
from torchvision import transforms

MINST_MEAN, MINST_STANDARD_DIV = 0.1307, 0.3081
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MINST_MEAN,), (MINST_STANDARD_DIV,))])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This function takes in an PIL Image and formats it to minsk format, then returns the image
# which has been processed.
def process_image(raw_image, new_size=28):
    image = format_image(raw_image)
    image = resize_and_center(image, new_size)
    return predict_image(image)


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
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    cnn_input = cnn_input.to(device)
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.to(device)  # Ensures model is on either CPU or GPU
    output = model(cnn_input)
    output = output.to(device)  # Does the same
    _, predicted = torch.max(output.data, 1)
    return predicted.sum().item()


# This function takes in an image, a figure size and a title and presents the image on a chart.
def show_image(image, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title: plt.title(title)
    plt.show()

