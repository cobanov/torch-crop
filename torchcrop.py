#!/usr/bin/env python

import argparse
import json
import math
import sys

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
        saturation_bias=0.2,
        saturation_brightness_max=0.9,
        saturation_brightness_min=0.05,
        saturation_threshold=0.4,
        saturation_weight=0.3,
        score_down_sample=8,
        skin_bias=0.01,
        skin_brightness_max=1,
        skin_brightness_min=0.2,
        skin_color=None,
        skin_threshold=0.8,
        skin_weight=1.8,
    ):
        self.detail_weight = detail_weight
        self.edge_radius = edge_radius
        self.edge_weight = edge_weight
        self.outside_importance = outside_importance
        self.rule_of_thirds = rule_of_thirds
        self.saturation_bias = saturation_bias
        self.saturation_brightness_max = saturation_brightness_max
        self.saturation_brightness_min = saturation_brightness_min
        self.saturation_threshold = saturation_threshold
        self.saturation_weight = saturation_weight
        self.score_down_sample = score_down_sample
        self.skin_bias = skin_bias
        self.skin_brightness_max = skin_brightness_max
        self.skin_brightness_min = skin_brightness_min
        self.skin_color = skin_color or self.DEFAULT_SKIN_COLOR
        self.skin_threshold = skin_threshold
        self.skin_weight = skin_weight

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
        cie_image = kornia.color.rgb_to_grayscale(image)

        # R=skin G=edge B=saturation
        edge_image = self.detect_edge(cie_image)
        skin_image = self.detect_skin(cie_image, image)
        saturation_image = self.detect_saturation(cie_image, image)
        print(
            "edge_image shape:",
            edge_image.shape,
            "skin_image shape:",
            skin_image.shape,
            "saturation_image shape:",
            saturation_image.shape,
        )
        analyse_image = torch.stack((edge_image, skin_image, saturation_image))
        print("Anaylse image shape: ", analyse_image.shape)

        del edge_image
        del skin_image
        del saturation_image

        score_image = analyse_image.clone()
        score_image = T.functional.resize(
            image,
            (
                int(math.ceil(image.shape[1] / self.score_down_sample)),
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
        scale = min(image.shape[1] / width, image.shape[2] / height)
        crop_width = int(math.floor(width * scale))
        crop_height = int(math.floor(height * scale))
        # img = 100x100, width = 95x95, scale = 100/95, 1/scale > min
        # don't set minscale smaller than 1/scale
        # -> don't pick crops that need upscaling
        min_scale = min(max_scale, max(1 / scale, min_scale))

        prescale_size = 1
        if prescale:
            prescale_size = 1 / scale / min_scale
            if prescale_size < 1:
                image = image.clone()
                image = T.functional.resize(
                    image,
                    (
                        int(image.shape[1] * prescale_size),
                        int(image.shape[2] * prescale_size),
                    ),
                    antialias=True,
                )

                crop_width = int(math.floor(crop_width * prescale_size))
                crop_height = int(math.floor(crop_height * prescale_size))
            else:
                prescale_size = 1

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
        image_width, image_height = image.shape[1], image.shape[2]
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

    def debug_crop(self, analyse_image, crop):
        debug_image = analyse_image.clone()
        debug_pixels = debug_image.getdata()  #! Might error
        debug_crop_image = Image.new(
            "RGBA",
            (int(math.floor(crop["width"])), int(math.floor(crop["height"]))),
            (255, 0, 0, 25),
        )
        ImageDraw.Draw(debug_crop_image).rectangle(
            ((0, 0), (crop["width"], crop["height"])), outline=(255, 0, 0)
        )

        for y in range(analyse_image.size[1]):  # height
            for x in range(analyse_image.size[0]):  # width
                p = y * analyse_image.size[0] + x
                importance = self.importance(crop, x, y)
                if importance > 0:
                    debug_pixels.putpixel(
                        (x, y),
                        (
                            debug_pixels[p][0],
                            int(debug_pixels[p][1] + importance * 32),
                            debug_pixels[p][2],
                        ),
                    )
                elif importance < 0:
                    debug_pixels.putpixel(
                        (x, y),
                        (
                            int(debug_pixels[p][0] + importance * -64),
                            debug_pixels[p][1],
                            debug_pixels[p][2],
                        ),
                    )
        debug_image.paste(
            debug_crop_image, (crop["x"], crop["y"]), debug_crop_image.split()[3]
        )
        return debug_image

    def detect_edge(self, cie_image):
        edge_image = cie_image.clone() / 255
        edge_image.unsqueeze_(0)
        magnitude, edges = kornia.filters.canny(edge_image)
        print(edges.shape)
        return edges[0]

    def detect_saturation(self, cie_image, source_image):
        threshold = self.saturation_threshold
        saturation_data = saturation(source_image)
        saturation_data.unsqueeze_(0)  #! Check that is True
        mask = (
            (saturation_data > threshold)
            & (cie_image >= self.saturation_brightness_min * 255)
            & (cie_image <= self.saturation_brightness_max * 255)
        )

        saturation_data[~mask] = 0
        saturation_data[mask] = (saturation_data[mask] - threshold) * (
            255 / (1 - threshold)
        )

        return saturation_data  # Image.fromarray(saturation_data.astype('uint8'))

    def detect_skin(self, cie_image, source_image):
        r, g, b = split(source_image)

        rd = torch.ones_like(r) * -self.skin_color[0]
        gd = torch.ones_like(g) * -self.skin_color[1]
        bd = torch.ones_like(b) * -self.skin_color[2]

        mag = torch.sqrt(r * r + g * g + b * b)
        mask = ~(abs(mag) < 1e-6)
        rd[mask] = r[mask] / mag[mask] - self.skin_color[0]
        gd[mask] = g[mask] / mag[mask] - self.skin_color[1]
        bd[mask] = b[mask] / mag[mask] - self.skin_color[2]

        skin = 1 - torch.sqrt(rd * rd + gd * gd + bd * bd)
        mask = (
            (skin > self.skin_threshold)
            & (cie_image >= self.skin_brightness_min * 255)
            & (cie_image <= self.skin_brightness_max * 255)
        )
        skin_data = (skin - self.skin_threshold) * (255 / (1 - self.skin_threshold))
        skin_data.unsqueeze_(0)  #! Check this is correct
        skin_data[~mask] = 0

        return skin_data  #  Image.fromarray(skin_data.astype('uint8'))

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
            "saturation": 0,
            "skin": 0,
            "total": 0,
        }
        # target_data = target_image.getdata()
        target_data = target_image.ravel().reshape(3, -1).T
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
                detail = target_data[p][1] / 255
                score["skin"] += (
                    target_data[p][0] / 255 * (detail + self.skin_bias) * importance
                )
                score["detail"] += detail * importance
                score["saturation"] += (
                    target_data[p][2]
                    / 255
                    * (detail + self.saturation_bias)
                    * importance
                )
        score["total"] = (
            score["detail"] * self.detail_weight
            + score["skin"] * self.skin_weight
            + score["saturation"] * self.saturation_weight
        ) / (crop["width"] * crop["height"])
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
    image = torchvision.io.read_image("./test.jpg")
    image = image.unsqueeze(0)
    print(image[0].shape)
    cropper = SmartCrop()

    cie_image = kornia.color.rgb_to_grayscale(image)[0]

    edge_image = cropper.detect_saturation(cie_image, image)
    print(edge_image)
    transform = T.ToPILImage()(edge_image)
    transform.show()


def main():
    options = parse_argument()
    image = torchvision.io.read_image(options.inputfile)
    # if image.mode != 'RGB' and image.mode != 'RGBA':
    #     sys.stderr.write("{1} convert from mode='{0}' to mode='RGB'\n".format(
    #         image.mode, options.inputfile))
    #     new_image = Image.new('RGB', image.size)
    #     new_image.paste(image)
    #     image = new_image

    cropper = SmartCrop()
    result = cropper.crop(
        image, width=100, height=int(options.height / options.width * 100)
    )

    box = (
        result["top_crop"]["x"],
        result["top_crop"]["y"],
        result["top_crop"]["width"] + result["top_crop"]["x"],
        result["top_crop"]["height"] + result["top_crop"]["y"],
    )
    print('BOX COORDINATES: ', box)

    if options.debug_file:
        analyse_image = result.pop("analyse_image")
        cropper.debug_crop(analyse_image, result["top_crop"]).save(options.debug_file)
        print(json.dumps(result))


    print('TYPE AND SHAPE', type(image), image.shape)

    image = T.ToPILImage()(image) #.crop(box).save(options.outputfile)


    cropped_image = image.crop(box)
    cropped_image.thumbnail((options.width, options.height), Image.ANTIALIAS)
    cropped_image.save(options.outputfile, "JPEG", quality=90)


if __name__ == "__main__":
    main()
