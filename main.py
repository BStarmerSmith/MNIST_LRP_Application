import src.data_loader as loader
import os
import argparse
import sys
from src.cnn import CNN
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--Train", help="Train the Network", action='store_true')
    parser.add_argument("-te", "--Test", help="Test the Network", action='store_true')
    args = parser.parse_args()
    if not len(sys.argv) > 1:
        print("Please use the flags -tr for training or -te for testing.")
    if args.Train:
        trainNetwork()
    if args.Test:
        testAllImages()