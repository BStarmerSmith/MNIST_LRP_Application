import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

# This function takes in a PIL image, converts it to a tensor and predicts what the image
# will be, it then passes the model and img_tensor to the LRP function which returns the
# relevancy of the input at all layers, and presents them.
def preform_lrp(image):
    model = torch.load(os.path.join(MODEL_DIRECTORY, MODEL_FILENAME))
    model.to(device)
    model.eval()
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    R = e_lrp(model, image_tensor)
    relevance = process_array([("Linear Layer", R[6][0]), ("MaxPool-2", R[5][0]), ("Conv2d-2", R[3][0]),
                               ("MaxPool-1", R[2][0]), ("Conv2d-1", R[0][0])])
    plot_images(image_tensor, relevance, str(predict_image(image)))


def process_array(arr):
    output = []
    for lable, element in arr:
        output.append((lable, np.array(element.cpu()).sum(axis=0)))
    return output


# This function takes in the model of the Network and an image_tensor. This is what
# The LRP algorithm is applied to, it converts all layers to Conv2D then preforms all
# forward passes so we can get the relevancy of the network at all layers.
# This function returns the relevance [R] of all layers within the network (except dropout).
def e_lrp(model, img_tensor):
    # Gets all the layers
    layers = list(model._modules['layer1']) + list(model._modules['layer2']) \
             + toconv([model._modules['fc1'], model._modules['fc2']])

    L = len(layers)
    A = [img_tensor]+[None]*L
    for l in range(L): # Preforms the forward pass for each layer
        A[l+1] = layers[l].forward(A[l].to(device))

    # Used to get the predicted output of the network
    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))
    T = np.abs(np.array(T)) * 0
    T[index] = 1
    T = torch.FloatTensor(T)

    R = [None] * L + [(A[-1].cpu() * T).data + 1e-6]
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # LRP paper says to turn MaxPool layers to AvgPoollayers
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        # We only preform LRP on Conv and AvgPool layers
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            if l <= 2:
                rho = lambda p: p + 0.25*p.clamp(min=0)
                incr = lambda z: z+1e-9
            if 3 <= l <= 5:
                rho = lambda p: p
                incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 6:
                rho = lambda p: p
                incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1].to(device)/z).data                   # step 2
            (z*s).sum().backward(); c = A[l].grad            # step 3
            R[l] = (A[l]*c).data                             # step 4
        else:
            R[l] = R[l+1]

    # Here is where we get the relevancy at layer 0
    mean, std = 0.1307, 0.3081  # mean and std of MNIST
    A[0] = (A[0].data).requires_grad_(True)
    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0].to(device)) + 1e-9                               # step 1 (a)
    z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb.to(device))   # step 1 (b)
    z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb.to(device))   # step 1 (c)
    s = (R[1] / z).data                                                         # step 2
    (z*s).sum().backward()                                                      # step 3
    c, cp, cm = A[0].grad, lb.grad, hb.grad
    R[0] = (A[0] * c + lb * cp + hb * cm).data
    return R


# This function takes a list of Linear layers and converts them to Conv2D layers.
# The layer at index 0 needs to be specifically formatted to deal with the adjustment
# of a 2d network to a 1d network.
def toconv(layers):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            newlayer = None
            if i == 0:
                m,n = 64,layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m,n,7)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m,n,1)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,1,1))
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


def plot_images(init_img, R, predicted_val):
    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 2
    i = 2 # As we are adding the input image first.
    fig.add_subplot(rows, columns, 1).set_title("Input Image")
    plt.axis('off')
    plt.imshow(init_img.reshape(28, 28), cmap='gray')
    for label, r in R:
        b = 10 * ((np.abs(r) ** 3.0).mean() ** (1.0/3))
        my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
        my_cmap[:, 0:3] *= 0.85
        my_cmap = ListedColormap(my_cmap)
        fig.add_subplot(rows, columns, i).set_title(label)
        plt.axis('off')
        plt.imshow(r, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
        i = i + 1
    plt.figtext(0.5, 0.1, "Predicted value of Network is {}".format(predicted_val), ha="center", fontsize=18,
                bbox={"facecolor":"purple", "alpha":0.5, "pad":5})

    plt.show()
