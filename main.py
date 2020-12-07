import src.data_loader as loader
import os
from src.cnn import CNN
from src.variables import *
from PIL import Image

def trainNetwork():
    model = CNN()
    model.trainModel()
    model.testModel()
    model.saveModel()

def testAllImages(path='data\\MyImages'):
    images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for img_dir in images:
        image = Image.open(img_dir)
        image = loader.formatImage(image)
        image = loader.resizeAndCenter(image)
        print(loader.predictImage(image))

if __name__ == "__main__":
    #train_network()
    testAllImages()