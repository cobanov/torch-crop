#!/usr/bin/env python

import argparse
import json
import math
import sys
import pickle

import torch
import kornia
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw


def split(image):
    print("Split shape:", image.shape)
    r, g, b = image[0, :, :], image[1, :, :], image[2, :, :]
    return r, g, b


def saturation(image):
    r, g, b = split(image)
    maximum = torch.maximum(torch.maximum(r, g), b)
    minimum = torch.minimum(torch.minimum(r, g), b)

    s = (maximum + minimum) / 255  # [0.0; 1.0]
    d = (maximum - minimum) / 255  # [0.0; 1.0]
    d[maximum == minimum] = 0  # if maximum == minimum:
    s[maximum == minimum] = 1  # -> saturation = 0 / 1 = 0
    mask = s > 1
    s[mask] = 2 - d[mask]
    return d / s  # [0.0; 1.0]


def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = ((x + 2 / 3) % 2 * 0.5 - 0.5) * 16
    return max(1 - x * x, 0)


class SmartCrop(object):

    DEFAULT_SKIN_COLOR = [0.78, 0.57, 0.44]

    def __init__(
        self,
        detail_weight=0.2,
        edge_radius=0.4,
        edge_weight=-20,
        outside_importance=-0.5,
        rule_of_thirds=True,
        score_down_sample=8,
    ):
        self.detail_weight = detail_weight
        self.edge_radius = edge_radius
        self.edge_weight = edge_weight
        self.outside_importance = outside_importance
        self.rule_of_thirds = rule_of_thirds
        self.score_down_sample = score_down_sample

    def analyse(
        self,
        image,
        crop_width,
        crop_height,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        """
        Analyze image and return some suggestions of crops (coordinates).
        This implementation / algorithm is really slow for large images.
        Use `crop()` which is pre-scaling the image before analyzing it.
        """
        # T.ToPILImage()(image).show()
        normalized_image = image / 255
        cie_image = kornia.color.rgb_to_grayscale(normalized_image)
        # T.ToPILImage()(cie_image[0]).show()
        print("cie_image:", cie_image.shape)

        # R=skin G=edge B=saturation
        edge_image = self.detect_edge(cie_image)

        analyse_image = edge_image.clone()
        # T.ToPILImage()(analyse_image[0]).show()
        print("Anaylse image shape: ", analyse_image.shape)

        del edge_image

        score_image = analyse_image.clone()
        print("Score image shape: ", score_image.shape)
        score_image = T.functional.resize(
            score_image,
            (
                int(math.ceil(image.shape[3] / self.score_down_sample)),
                int(math.ceil(image.shape[2] / self.score_down_sample)),
            ),
            antialias=True,
        )

        top_crop = None
        top_score = -sys.maxsize

        crops = self.crops(
            image,
            crop_width,
            crop_height,
            max_scale=max_scale,
            min_scale=min_scale,
            scale_step=scale_step,
            step=step,
        )

        for crop in crops:
            crop["score"] = self.score(score_image, crop)
            if crop["score"]["total"] > top_score:
                top_crop = crop
                top_score = crop["score"]["total"]

        return {"analyse_image": analyse_image, "crops": crops, "top_crop": top_crop}

    def crop(
        self,
        image,
        width,
        height,
        prescale=True,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        """Not yet fully cleaned from https://github.com/hhatto/smartcrop.py."""
        print("crop:", image.shape)
        prescale_size = 1
        crop_width = 500
        crop_height = 500


        result = self.analyse(
            image,
            crop_width=crop_width,
            crop_height=crop_height,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
            step=step,
        )

        for i in range(len(result["crops"])):
            crop = result["crops"][i]
            crop["x"] = int(math.floor(crop["x"] / prescale_size))
            crop["y"] = int(math.floor(crop["y"] / prescale_size))
            crop["width"] = int(math.floor(crop["width"] / prescale_size))
            crop["height"] = int(math.floor(crop["height"] / prescale_size))
            result["crops"][i] = crop

        return result

    def crops(
        self,
        image,
        crop_width,
        crop_height,
        max_scale=1,
        min_scale=0.9,
        scale_step=0.1,
        step=8,
    ):
        image_width, image_height = (
            image.shape[2],
            image.shape[3],
        )  # pytorch dims are (channels, height, width)
        print("image_width:", image_width, "image_height:", image_height)
        crops = []
        for scale in (
            i / 100
            for i in range(
                int(max_scale * 100),
                int((min_scale - scale_step) * 100),
                -int(scale_step * 100),
            )
        ):
            for y in range(0, image_height, step):
                if not (y + crop_height * scale <= image_height):
                    break
                for x in range(0, image_width, step):
                    if not (x + crop_width * scale <= image_width):
                        break
                    crops.append(
                        {
                            "x": x,
                            "y": y,
                            "width": crop_width * scale,
                            "height": crop_height * scale,
                        }
                    )
        if not crops:
            raise ValueError(locals())
        return crops

    def detect_edge(self, cie_image):
        # T.ToPILImage()(cie_image).show()
        edge_image = cie_image.clone()
        # edge_image.unsqueeze_(0)
        _, edges = kornia.filters.canny(edge_image, kernel_size=(3, 3))
        # edges = kornia.filters.sobel(
        #     edge_image,
        # )
        # T.ToPILImage()(edges[0]).show()
        print(edges.shape)
        return edges

    def importance(self, crop, x, y):
        if (
            crop["x"] > x
            or x >= crop["x"] + crop["width"]
            or crop["y"] > y
            or y >= crop["y"] + crop["height"]
        ):
            return self.outside_importance

        x = (x - crop["x"]) / crop["width"]
        y = (y - crop["y"]) / crop["height"]
        px, py = abs(0.5 - x) * 2, abs(0.5 - y) * 2

        # distance from edge
        dx = max(px - 1 + self.edge_radius, 0)
        dy = max(py - 1 + self.edge_radius, 0)
        d = (dx * dx + dy * dy) * self.edge_weight
        s = 1.41 - math.sqrt(px * px + py * py)

        if self.rule_of_thirds:
            s += (max(0, s + d + 0.5) * 1.2) * (thirds(px) + thirds(py))

        return s + d

    def score(self, target_image, crop):
        score = {
            "detail": 0,
            "total": 0,
        }
        # target_data = target_image.getdata()
        target_data = target_image.ravel().reshape(3, -1).T #! Ravel icin batch processing hamlesi dusun
        target_width, target_height = target_image.shape[-2:]

        down_sample = self.score_down_sample
        inv_down_sample = 1 / down_sample
        target_width_down_sample = target_width * down_sample
        target_height_down_sample = target_height * down_sample

        for y in range(0, target_height_down_sample, down_sample):
            for x in range(0, target_width_down_sample, down_sample):
                p = int(
                    math.floor(y * inv_down_sample) * target_width
                    + math.floor(x * inv_down_sample)
                )
                importance = self.importance(crop, x, y)
                detail = target_data[p][1]

                score["detail"] += detail * importance

        score["total"] = (score["detail"] * self.detail_weight) / (
            crop["width"] * crop["height"]
        )
        return score


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", metavar="INPUT_FILE", help="Input image file")
    parser.add_argument("outputfile", metavar="OUTPUT_FILE", help="Output image file")
    parser.add_argument(
        "--debug-file", metavar="DEBUG_FILE", help="Debugging image file"
    )
    parser.add_argument(
        "--width", dest="width", type=int, default=100, help="Crop width"
    )
    parser.add_argument(
        "--height", dest="height", type=int, default=100, help="Crop height"
    )
    return parser.parse_args()


def main():
    # options = parse_argument()
    # image = torchvision.io.read_image(options.inputfile)

    # load data.pkl
    with open("data.pkl", "rb") as f:
        image = pickle.load(f)
        print("Init image shape: ", image.shape)


    cropper = SmartCrop()
    result = cropper.crop(image, width=500, height=int(1024 / 1024 * 500)) #! Dont forget to back

    box = (
        result["top_crop"]["x"],
        result["top_crop"]["y"],
        result["top_crop"]["width"] + result["top_crop"]["x"],
        result["top_crop"]["height"] + result["top_crop"]["y"],
    )
    print("BOX COORDINATES: ", box)

    # if options.debug_file:
    #     analyse_image = result.pop("analyse_image")
    #     cropper.debug_crop(analyse_image, result["top_crop"]).save(options.debug_file)
    #     print(json.dumps(result))

    print("TYPE AND SHAPE", type(image), image.shape)

    image = T.ToPILImage()(image[7])  # .crop(box).save(options.outputfile)

    cropped_image = image.crop(box)
    cropped_image.show()
    # cropped_image.thumbnail((options.width, options.height), Image.ANTIALIAS)
    # cropped_image.save(options.outputfile, "JPEG", quality=90)


if __name__ == "__main__":
    main()
