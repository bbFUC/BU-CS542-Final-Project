from pathlib import Path
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter 


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, batch_size, csv_writer=None, save_dir=None, writer1=None, writer2=None, save_freq=1):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.writer1 = writer1
        self.writer2 = writer2
        self.csv_writer = csv_writer

    def _iteration(self, data_loader, epoch, is_train=False):
        loop_loss = []
        accuracy = [0]
        accuracyK = [0]
        image_name_list = data_loader.dataset.imgs
        count_test = 0
        for data, target in tqdm(data_loader, ncols=80):          
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            # get the topK prediction of the test images
            # the first argument means the K of topK
            output_topk = output.data.topk(3,1,True,True)[1]
            for index_list in output_topk:
                image_id_raw = image_name_list[count_test][0]
                image_id = str(image_id_raw.split('/')[-1].replace('.png',''))
                count_test += 1
                self.csv_writer.write(image_id + ',')
                for index in index_list:
                    prediction_class = str(data_loader.dataset.classes[index])
                    self.csv_writer.write(prediction_class + ' ')
                self.csv_writer.write('\n')
        
        index2category = open('./index2category.txt', 'a')
        for i in range(0, 340):
            index2category.write(str(data_loader.dataset.classes[i]) + '\n')
        index2category.close()

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
