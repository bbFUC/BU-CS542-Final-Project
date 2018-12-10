from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime


from resnet import resnet50
from utils import Trainer
from tensorboardX import SummaryWriter 


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.Resize(299), #transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
    data_augmentation = [#transforms.RandomSizedCrop(224),
                         transforms.RandomHorizontalFlip(), ]

    traindir = str(Path(root) / "train")
    valdir = str(Path(root) / "val")
    train = datasets.ImageFolder(traindir, transforms.Compose(data_augmentation + to_normalized_tensor))
    val = datasets.ImageFolder(valdir, transforms.Compose(to_normalized_tensor))
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        val, batch_size=batch_size, shuffle=True, num_workers=8)
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


def main(batch_size, root):
    #####################################################################
    "The implementation of tensorboardX and topK accuracy is in utils.py"
    #####################################################################

    # get checkpoint information
    checkpoint_newest = get_checkPoint("./checkpoint/")

    #TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # write log and visualize the losses of batches of training and testing
    TIMESTAMP = ""
    writer1 = SummaryWriter('./tensorboard_log/batch/'+TIMESTAMP)
    # write log and visualize the accuracy of batches of training and testing
    writer2 = SummaryWriter('./tensorboard_log/epoch/'+TIMESTAMP)

    train_loader, test_loader = get_dataloader(batch_size, root)
    gpus = list(range(torch.cuda.device_count()))

    # initialize your net/optimizer
    inception_v4 = nn.DataParallel(resnet50(num_classes=345),
                                device_ids=gpus)
    optimizer = optim.SGD(params=inception_v4.parameters(), lr=0.6 / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)

    # No existed checkpoint
    if checkpoint_newest == 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        trainer = Trainer(inception_v4, optimizer, F.cross_entropy, save_dir="./checkpoint/", writer1=writer1, writer2=writer2, save_freq=1)
        trainer.loop(100, train_loader, test_loader, 1, scheduler)
    # load existed checkpoint
    else:
        print("The path of the pretrained model %s" %checkpoint_newest)
        print("load pretrained model......")
        checkpoint = torch.load(checkpoint_newest)
        inception_v4.load_state_dict(checkpoint['weight'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=checkpoint['epoch'])
        print("The current epoch is %d" %checkpoint['epoch'])
        trainer = Trainer(inception_v4, optimizer, F.cross_entropy, save_dir="./checkpoint/", writer1=writer1, writer2=writer2, save_freq=1)
        trainer.loop(100, train_loader, test_loader, checkpoint['epoch']+1, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=4, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root)
