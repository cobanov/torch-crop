import torch
import torchvision
import kornia
import torchvision.transforms as T
from PIL import Image



############ DETECT EDGE
def detect_edge(image):
    image = image.unsqueeze_(0)/255.0
    magnitude, edges = kornia.filters.canny(image)
    return magnitude, edges

########## DETECT SATURATION

def split(image):
    r, g, b = image[0, :, :],image[1, :, :], image[2, :, :]
    return r, g, b


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


def detect_saturation(image):
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

###### DETECT SKIN

def detect_skin(image):
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



image = torchvision.io.read_image('./test.jpeg')

# magnitude, edges = detect_edge(image)
# sat = detect_saturation(image)
skin = detect_skin(image)
print(skin)




# transform = T.ToPILImage()(edges[0])
# transform.show()