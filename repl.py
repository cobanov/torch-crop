

import argparse
import json
import math
import sys

import torch
import kornia
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
from PIL.ImageFilter import Kernel

def main():
    score_down_sample=8
    image = torchvision.io.read_image("./test.jpg")
    print(image)

    target_image = Image.open("./test.jpg")
    target_data = target_image.getdata()
    for data in target_data:
        print(data)





if __name__ == "__main__":
    main()
