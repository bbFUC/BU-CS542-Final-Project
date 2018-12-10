from pathlib import Path
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter 


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, writer1=None, writer2=None, save_freq=1):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.writer1 = writer1
        self.writer2 = writer2

    def _iteration(self, data_loader, epoch, is_train=True):
        loop_loss = []
        accuracy = [0]
        accuracyK = [0]
        count = (epoch-1)*len(data_loader)  # count is the number of current batch
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            # get the topK prediction of the test images
            # the first argument means the K of topK
            output_topk = output.data.topk(3,1,True,True)[1]
            accuracyK_count = 0.0   # used to count the topK accuracy
            # MAP@3 accuracy
            for (i,j) in zip(target.data, output_topk):
                tem = 1
                for k in j:
                    if k == i:
                        accuracyK_count += 1 / tem
                        break
                    else:
                        tem += 1

            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            accuracyK.append(accuracyK_count)
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # write and visualize the batch of train&test loss&accuracy in tensorboard
            if is_train:
                self.writer1.add_scalar('train_batch_loss', loss, count)
                self.writer1.add_scalars('train_batch_accuracy&accuracyK', {"accuracy":sum(accuracy) / len(data_loader.dataset), "accuracyK":sum(accuracyK) / len(data_loader.dataset)}, count)
            count+=1
        mode = "train" if is_train else "test"
        print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}/accuracyK: {sum(accuracyK) / len(data_loader.dataset):.2%}")

        # write and visualize the epoch of train&test loss&accuracy in tensorboard
        if is_train:
            self.writer2.add_scalars('epoch_loss', {"train_loss":sum(loop_loss)}, epoch)
            self.writer2.add_scalars('epoch_accuracy&accuracyK', {"train_accuracy":sum(accuracy) / len(data_loader.dataset), "train_accuracyK":sum(accuracyK) / len(data_loader.dataset)}, epoch)
        else:
            self.writer2.add_scalars('epoch_loss', {"test_loss":sum(loop_loss)}, epoch)
            self.writer2.add_scalars('epoch_accuracy&accuracyK', {"test_accuracy":sum(accuracy) / len(data_loader.dataset), "test_accuracyK":sum(accuracyK) / len(data_loader.dataset)}, epoch)
        return loop_loss, accuracy, accuracyK

    def train(self, data_loader, epoch):
        self.model.train()
        with torch.enable_grad():
            loss, correct, correctK = self._iteration(data_loader, epoch)

    def test(self, data_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            loss, correct, correctK = self._iteration(data_loader, epoch, is_train=False)

    def loop(self, epochs, train_data, test_data, current_epoch, scheduler=None, test_mode=0):
        for ep in range(current_epoch, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            if test_mode == 0:
                self.train(train_data, ep)
                self.test(test_data, ep)
                if ep % self.save_freq == 0:
                    self.save(ep)
            else:
                self.test(test_data, ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "weight": self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
