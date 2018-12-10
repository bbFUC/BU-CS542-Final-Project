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
from inception_v4 import inceptionv4
from utils import Trainer
from tensorboardX import SummaryWriter 


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.Resize(299), #transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
    data_augmentation = [#transforms.RandomSizedCrop(224),
                         transforms.RandomHorizontalFlip(), ]

    valdir = str(Path(root) / "val")
    val = datasets.ImageFolder(valdir, transforms.Compose(to_normalized_tensor))
    test_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader


# root is the dir of checkpoints
# num is which checkpoint you want to load, default is the newest
def get_checkPoint(root, num=None):
    pattern = "model_epoch_1.pth"
    dir = os.listdir(root)
    # load the newest checkpoint
    if num == None:
        if dir == []:
            return 0
        newest = 1
        for checkpoint in dir:
            checkpoint = checkpoint.replace("model_epoch_", "")
            checkpoint = checkpoint.replace(".pth", "")
            if int(checkpoint) > newest:
                newest = int(checkpoint)
        pattern = pattern.replace("1", str(newest))
    
    # load all checkpoints from start to end
    elif num == 0:
        checkpoints = []
        for checkpoint in dir:
            checkpoints.append(root+checkpoint)
        return checkpoints

    else:
        pattern = pattern.replace("1", str(num))
    
    if num == None:
        num = newest
    abPath = os.path.abspath(root + pattern)
    print(abPath)
    return abPath


def main(batch_size, root):
    # torchvision.datasets.DataLoader()
    #####################################################################
    "The implementation of tensorboardX and topK accuracy is in utils.py"
    #####################################################################

    # test mode: 1=open, 0=close
    test_mode = 1

    # get checkpoint information
    checkpoint_newest = get_checkPoint("./checkpoint/", 1)
    test_loader = get_dataloader(batch_size, root)
    gpus = list(range(torch.cuda.device_count()))

    # initialize your net/optimizer
    nameOfNet = "se_resnet_testResult.csv"
    se_resnet = nn.DataParallel(inceptionv4(num_classes=3),
                                device_ids=gpus)
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6 / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)

    # No existed checkpoint
    if checkpoint_newest == 0:
        print("-------------No checkpoint available!!!!--------------")
        
    # load existed checkpoint
    else:
        csv_path = "./" + nameOfNet
        csv_writer = open(csv_path, "w")
        csv_writer.write("key_id,word\n")
        checkpoint_newest_list = []
        if isinstance(checkpoint_newest, list) == False:
            checkpoint_newest_list.append(checkpoint_newest)
        else:
            checkpoint_newest_list = checkpoint_newest
        for checkpoint_path in checkpoint_newest_list:
            print("The path of the pretrained model %s" %checkpoint_path)
            print("load pretrained model......")
            checkpoint = torch.load(checkpoint_path)
            se_resnet.load_state_dict(checkpoint['weight'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
            print("The current epoch is %d" %checkpoint['epoch'])
            print("prepare to write the csv file for testing...")
            trainer = Trainer(se_resnet, optimizer, F.cross_entropy, batch_size, csv_writer, save_dir="./checkpoint/", save_freq=1)
            train_loader = None
            trainer.loop(checkpoint['epoch'], train_loader, test_loader, checkpoint['epoch'], scheduler, test_mode)



if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=16, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root)
