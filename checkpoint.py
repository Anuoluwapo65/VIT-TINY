import urllib.request

from module import os, np
import torch
from urllib.error import HTTPError, URLError
import pytorch_lightning as pl

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = "./data_path"
checkpoint_path = "./checkpoint_path"



base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/" 
pretrained_filename = "tutorial15/ViT.ckpt", "tutorial15/tensorboards/ViT/events.out.tfevents.ViT",
"tutorial5/tensorboards/ResNet/events.out.tfevents.resnet"

def configure_seed(seed:int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

configure_seed(42)
def pretrained_model( url:str = base_url, filename:str = pretrained_filename):
    for file in filename:
        file_path = os.path.join(checkpoint_path, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        urls_path = url + file
        if not os.path.isfile(file_path):
            print(f" downloading the {urls_path}")
            try:
                urllib.request.urlretrieve(urls_path, file_path)
            except (HTTPError, URLError, OSError) as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                    e,
                )

pretrained_model()
