import src.data_loader as loader
import os
import argparse
import sys
from src.cnn import CNN
from PIL import Image


def train_network():
    model = CNN()
    model.train_model()
    model.test_model()
    model.save_model()


def test_all_images(path='data\\MyImages'):
    images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for img_dir in images:
        image = Image.open(img_dir)
        print(loader.process_image(image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--Train", help="Train the Network", action='store_true')
    parser.add_argument("-te", "--Test", help="Test the Network", action='store_true')
    args = parser.parse_args()
    if not len(sys.argv) > 1:
        print("Please use the flags -tr for training or -te for testing.")
    if args.Train:
        train_network()
    if args.Test:
        test_all_images()

