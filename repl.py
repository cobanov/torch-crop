

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
    w, h = image.shape[-2:]
    print(w ,h)
    t_image = image.ravel().reshape(3, -1).T

    target_image = Image.open("./test.jpg")
    target_data = target_image.getdata()







if __name__ == "__main__":
    main()
