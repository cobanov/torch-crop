import torchvision
import torch
from torchvision import transforms
import kornia
import pickle




def load_dataset():
    data_path = 'images'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True
    )
    return train_loader

dataset = load_dataset()

# def detect_edge(self, image):
#     edge_image = image.clone()
#     expand_image = edge_image.unsqueeze_(0)/255.0
#     magnitude, edges = kornia.filters.canny(expand_image)
#     return edges[0, 0, :, :]

for batch_idx, (data, target) in enumerate(load_dataset()):
    print(data.shape)
    # pickle data
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

    # magnitude, edges = kornia.filters.canny(data)
    # for idx, image in enumerate(edges):
    #     torchvision.utils.save_image(image, f'./edges/edge_{batch_idx}_{idx}.png')
