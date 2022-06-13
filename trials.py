import torch
import torchvision
import kornia
import torchvision.transforms as T

def saturation(image):
    r, g, b = split(image)
    maximum = torch.maximum(torch.maximum(r,g),b)
    minumum = torch.minimum(torch.minimum(r,g),b)
    s = (maximum + minumum) / 255
    d = (maximum - minumum) / 255
    d[maximum == minumum] = 0
    s[maximum == minumum] = 1
    mask = s > 1
    s[mask] = 2 - d[mask]
    return d / s

def thirds(x):
    """gets value in the range of [0, 1] where 0 is the center of the pictures
    returns weight of rule of thirds [0, 1]"""
    x = ((x + 2 / 3) % 2 * 0.5 - 0.5) * 16
    return max(1 - x * x, 0)

def split(image):
    r, g, b = image[0, :, :],image[1, :, :], image[2, :, :]
    return r, g, b


class TorchCrop(object):
    def __init__(self, saturation_threshold=0.4):
        self.saturation_threshold = saturation_threshold

    # def crop(self, image):
    #     self.image = image
    #     self.image_array = self.image/255.0
    #     self.edges = self.detect_edge(self.image_array)

    def detect_edge(self, image):
        edge_image = image.clone()
        expand_image = edge_image.unsqueeze_(0)/255.0
        magnitude, edges = kornia.filters.canny(expand_image)
        return edges[0, 0, :, :]

    def detect_skin(self, image):
        skin_color = [0.78, 0.57, 0.44]
        r, g, b = split(image)
        grayscale_image = kornia.color.rgb_to_grayscale(image)[0]

        rd = torch.ones_like(r) * -skin_color[0]
        gd = torch.ones_like(g) * -skin_color[1]
        bd = torch.ones_like(b) * -skin_color[2]

        mag = torch.sqrt(r * r + g * g + b * b)
        mask = ~(abs(mag) < 1e-6)
        rd[mask] = r[mask] / mag[mask] - skin_color[0]
        gd[mask] = g[mask] / mag[mask] - skin_color[1]
        bd[mask] = b[mask] / mag[mask] - skin_color[2]

        skin = 1 - torch.sqrt(rd * rd + gd * gd + bd * bd)
        mask = (
            (skin > 0.8) &
            (grayscale_image >= 0.2 * 255) &
            (grayscale_image <= 1 * 255))

        skin_data = (skin - 0.8) * (255 / (1 - 0.8))
        skin_data[~mask] = 0

        return skin_data


    def detect_saturation(self, image):
        grayscale_image = kornia.color.rgb_to_grayscale(image)[0]

        threshold = 0.1
        saturation_data = saturation(image)
        mask = (
            (saturation_data > threshold) &
            (grayscale_image >= 0.05 * 255) &
            (grayscale_image <= 0.9 * 255))

        saturation_data[~mask] = 0
        saturation_data[mask] = (saturation_data[mask] - threshold) * (255 / (1 - threshold))

        return saturation_data # Image.fromarray(saturation_data.astype('uint8'))

    def analyse(self, image):
        edge_image = self.detect_edge(image)
        skin_image = self.detect_skin(image)
        saturation_image = self.detect_saturation(image)
        print('edges: ', edge_image.shape)
        print('edges: ', skin_image.shape)
        print('edges: ', saturation_image.shape)

        analyse_image = torch.stack((edge_image, skin_image, saturation_image))
        return analyse_image


def main():
    image = torchvision.io.read_image('./test.jpeg')
    crop = TorchCrop()
    analysed_image = crop.analyse(image)
    print(analysed_image.shape)



main()