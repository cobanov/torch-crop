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
    score_down_sample = 2
    image = (
        torchvision.io.read_image("./images/city/ec50k_00010013.jpg").unsqueeze(0) / 255
    )
    score_image = T.functional.resize(
        image,
        (183, 111),
        antialias=True,
    )
    cie_image = kornia.color.rgb_to_grayscale(score_image)
    T.ToPILImage()(cie_image[0]).show()
    _, edges = kornia.filters.canny(cie_image)
    T.ToPILImage()(edges[0]).show()

    # print(w ,h)
    # t_image = image.ravel().reshape(3, -1).T

    # target_image = Image.open("./test.jpg")
    # print(target_image.size)
    # target_data = target_image.getdata()


if __name__ == "__main__":
    main()
