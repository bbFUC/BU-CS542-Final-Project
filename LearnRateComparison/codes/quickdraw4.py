from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

from se_resnet import se_resnet50
from utils import Trainer
from tensorboardX import SummaryWriter 


import sys
import numpy as np
from PIL import Image

# Use a little wrapper to add some error
# handling to the image loading.
def robust_load(imgpath):
    try:
        img = Image.open(imgpath)
        return img.convert('RGB')
    except Exception as e:
        # Print a complaint to stderr.
        sys.stderr.write('\n**** ERROR **** Unable to open file: %s\n' % imgpath)
        sys.stderr.write('**** Exception message: %s\n' % e)
        sys.stderr.flush()
    # Return a 1 pixel PIL image.  This will cause no harm
    # during image loading and will get filtered out in the 
    # no_bad_imgs_collate function
    return Image.fromarray(np.zeros([3,1,1]),'RGB')
    
    
# Define the image size
RESIZE=224
# A custom collate_fn is used to filter out any image whose size
# is not RESIZExRESIZE. Images that failed to read properly will
# match this criteria.
# NOTE THIS WILL SOMETIMES RESULT IN A REDUCED BATCH SIZE!
# PyTorch expects that its data sets are pre-filtered to be correct
# and not contain erroneous images.
def no_bad_imgs_collate(batch):
    # If a tensor is outside of size RESIZExRESIZE then skip it!
    correct_size = torch.Size((3,RESIZE,RESIZE))
    filtered_batch = []
    for item in batch:
        if item[0].shape == correct_size:
            filtered_batch.append(item)
    # Hand off to the default collate function
    return torch.utils.data.dataloader.default_collate(filtered_batch)


# Get the SCC assigned number of cores if available,
# otherwise return 1
def get_nslots():
    if 'NSLOTS' in os.environ:
        return int(os.environ['NSLOTS'])
    return 1

def get_dataloader(batch_size, root):
    # To change the resized image size edit the RESIZE parameter above @ line 39
    to_normalized_tensor = [transforms.Resize(RESIZE), #transforms.CenterCrop(RESIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
    data_augmentation = [#transforms.RandomSizedCrop(224),
                         transforms.RandomHorizontalFlip(), ]

    traindir = str(Path(root) / "smalltrain")
    valdir = str(Path(root) / "val")
    train =  datasets.ImageFolder(traindir, transforms.Compose(data_augmentation + to_normalized_tensor),loader=robust_load) 
    val =  datasets.ImageFolder(valdir, transforms.Compose(to_normalized_tensor),loader=robust_load) 
    train_loader =  DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=get_nslots(),collate_fn=no_bad_imgs_collate)
    test_loader =  DataLoader(
        val, batch_size=batch_size, shuffle=True, num_workers=get_nslots(),collate_fn=no_bad_imgs_collate)
    return train_loader, test_loader


# root is the dir of checkpoints
# num is which checkpoint you want to load, default is the newest
def get_checkPoint(root, num=None):
    pattern = "model_epoch_1.pth"
    if num == None:
        dir = os.listdir(root)
        if dir == []:
            return 0
        newest = 1
        for checkpoint in dir:
            checkpoint = checkpoint.replace("model_epoch_", "")
            checkpoint = checkpoint.replace(".pth", "")
            if int(checkpoint) > newest:
                newest = int(checkpoint)
        pattern = pattern.replace("1", str(newest))
        
    else:
        pattern = pattern.replace("1", num)
    
    if num == None:
        num = newest
    return root + pattern


def main(batch_size, root, lrate):
    #####################################################################
    "The implementation of tensorboardX and topK accuracy is in utils.py"
    #####################################################################

    # get checkpoint information
    checkpoint_newest = get_checkPoint("./lr"+str(lrate)+"/checkpoint/")

    #TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # write log and visualize the losses of batches of training and testing
    TIMESTAMP = ""
    writer1 = SummaryWriter('./lr'+str(lrate)+'/tensorboard_log/batch/'+TIMESTAMP)
    # write log and visualize the accuracy of batches of training and testing
    writer2 = SummaryWriter('./lr'+str(lrate)+'/tensorboard_log/epoch/'+TIMESTAMP)

    train_loader, test_loader = get_dataloader(batch_size, root)
    gpus = list(range(torch.cuda.device_count()))

    # initialize your net/optimizer
    seresnet50 = nn.DataParallel(se_resnet50(num_classes=340),
                                device_ids=gpus)
    optimizer = optim.SGD(params=seresnet50.parameters(), lr=lrate / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)

    # No existed checkpoint
    if checkpoint_newest == 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        trainer = Trainer(seresnet50, optimizer, F.cross_entropy, save_dir="./lr"+str(lrate)+"/checkpoint/", writer1=writer1, writer2=writer2, save_freq=1)
        trainer.loop(50, train_loader, test_loader, 1, scheduler)
    # load existed checkpoint
    else:
        print("The path of the pretrained model %s" %checkpoint_newest)
        print("load pretrained model......")
        checkpoint = torch.load(checkpoint_newest)
        seresnet50.load_state_dict(checkpoint['weight'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=checkpoint['epoch'])
        print("The current epoch is %d" %checkpoint['epoch'])
        trainer = Trainer(seresnet50, optimizer, F.cross_entropy, save_dir="./lr"+str(lrate)+"/checkpoint/", writer1=writer1, writer2=writer2, save_freq=1)
        trainer.loop(100, train_loader, test_loader, checkpoint['epoch']+1, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--lrate", default=0.1, type=float)
    args = p.parse_args()
    main(args.batch_size, args.root, args.lrate)
