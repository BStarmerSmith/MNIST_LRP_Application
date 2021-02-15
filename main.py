import src.data_loader as loader
import os
import argparse
import sys
import random
from src.cnn import CNN
from PIL import Image

output_path = 'data/Output'

def train_network():
    model = CNN()
    model.train_model()
    model.test_model()
    model.save_model()


# This function is used to test all the images in a specifc folder. Its by default set to the path of
# Images I provided for testing.
def process_images(path='data\\MyImages'):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for img in images:
        img_dir = os.path.join(path, img)
        out_dir = os.path.join(output_path, img)
        image = Image.open(img_dir)
        loader.process_image(image, out_dir)
        print("Done image {}".format(img))


# This function is used to test one random image I created.
def process_image(path='data\\MyImages'):
    img_choice = random.choice([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    img_dir = os.path.join(path, img_choice)
    out_dir = os.path.join(output_path, img_choice)
    image = Image.open(img_dir)
    loader.process_image(image, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--Train", help="Train the Network", action='store_true')
    parser.add_argument("-te", "--Test", help="Test the Network", action='store_true')
    args = parser.parse_args()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not len(sys.argv) > 1:
        print("Please use the flags -tr for training or -te for testing.")
    if args.Train:
        train_network()
    if args.Test:
        process_images()

