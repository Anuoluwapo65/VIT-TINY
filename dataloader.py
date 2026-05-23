import torch    
import torchvision   
from torchvision.datasets import CIFAR10 
from torchvision import transforms  
from torch.utils.data import DataLoader, random_split 
import numpy  as np      
from checkpoint import data_path     
import torch.utils.data as data
import matplotlib.pyplot as plt
train_dataset = CIFAR10(root = data_path, train = True, download = True)

img_mean =  (train_dataset.data.mean(axis=(0,1,2)) / 255.0)
img_std = (train_dataset.data.std(axis = (0,1,2))/ 255.0)
print(img_mean)
print(img_std)

train_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(p = 0.5),
                                       transforms.RandomVerticalFlip(p = 0.5),
                                       transforms.RandomResizedCrop((32, 32), scale = (0.2, 0.8), ratio = (0.4, 0.6)),
                                       transforms.Normalize(mean = img_mean, std = img_std)])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean = img_mean, std = img_std)])



train_datasets = CIFAR10(root = data_path, train = True, download = True, transform = train_transform)
val_datasets = CIFAR10(root = data_path, train = True, download = True, transform = test_transform)
test_datasets = CIFAR10(root = data_path, train = False, download = True, transform = test_transform)

train_size = int(0.8 * len(train_datasets))
val_size = len(train_datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_datasets, [train_size, val_size])
train_dataloader = data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 0, pin_memory = True)
val_dataloader = data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)
test_dataloader = data.DataLoader(test_datasets, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)

def img_patches(x, patch_size, return_channels = True):
  batch_size, channels, height, width = x.shape   
  x = x.reshape(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
  x = x.permute(0, 2, 4, 1, 3, 5)
  x = x.flatten(1, 2)
  x = x.flatten(2, 4)
  return x


if __name__ == "__main__":
  num_images = 10
  img_batch = [val_datasets[idx][0] for idx in range(num_images)]
  img_batch = torch.stack(img_batch, dim = 0)
  grid = torchvision.utils.make_grid(img_batch, nrow = 4, padding = 2, pad_value = 0.5, normalize = True)
  grid_img = grid.permute(1, 2, 0)

  plt.figure(figsize = (8, 10))
  plt.imshow(grid_img.numpy())
  plt.title("sample images from the CIFAR10 dataset")
  plt.axis('off')
  plt.show()

  img_patch = img_patches(img_batch, patch_size = 4, return_channels = False)
  fig, ax = plt.subplots(img_batch.shape[0], 1, figsize = (7, 9))
  plt.suptitle("Image patches of the CIFAR10 dataset")
  for i in range(img_batch.shape[0]):
    img_paths = torchvision.utils.make_grid(img_patch[i], nrow = 4, padding = 4, pad_value = 0.5, normalize = True)
    img_paths = img_paths.permute(1, 2, 0)
    ax[i].imshow(img_paths.numpy())
    ax[i].set_title(f"Patch {i+1}")
    ax[i].axis('off')
  plt.tight_layout()
  plt.show()

