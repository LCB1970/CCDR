import copy
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset_helper.data_reader import *
from segmentation import UNet
from loss import triplet_loss
from augmentation.tsrm import TSRM
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', default=4, type=int)
parser.add_argument('--data_path', default='data')
parser.add_argument('--save_weight_path', default=None)
parser.add_argument('--save_png_path', default=None)
parser.add_argument('-f', '--format', default='/*.png')

parser.add_argument('-c', '--channel', default=3, type=int)
parser.add_argument('--classes', default=3, type=int)
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-bs', '--batch_size', default=6, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float)
parser.add_argument('--kernel_size_list', default=[1, 3, 5, 7], type=list)

parser.add_argument('--L_max', default=10, type=int)
parser.add_argument('--gamma_std', default=1, type=float)
parser.add_argument('--beta_std', default=1, type=float)
parser.add_argument('--xi', default=0.5, type=float, help='margin value')
args = parser.parse_args()

device = torch.device("cuda")
data_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

dataset = MyDataset(
    data_root=args.data_path,
    dataset_name='paris',
    transform=data_transform,
    img_format=args.format,
    label_format=args.format
)
data_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=True, drop_last=False, num_workers=0)

model = UNet(band=args.channel, num_classes=args.classes, is_ccdr=True)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, data_loader, num_epochs):
    model = nn.DataParallel(model, device_ids=[i for i in range(args.gpu_number)]).to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all, train_acc_all = [], []
    val_loss_all, val_acc_all = [], []
    since = time.time()

    for epoch in range(num_epochs):
        if (epoch + 1) % 100 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss, seg_loss, align_loss, train_num, val_loss, val_num = 0, 0, 0, 0, 0, 0
        model.train()
        model.to(device)

        for step, (x, y) in enumerate(data_loader):
            kf = KFold(n_splits=2, shuffle=True)
            for train_index, val_index in kf.split(x):
                optimizer.zero_grad()
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]
                model.train()
                x_train, y_train = x_train.float().to(device), y_train.long().to(device)
                bs_class_num_count = []
                for bi in range(args.batch_size):
                    class_num_count = []
                    for ci in range(args.classes):
                        temp = torch.sum(y_train[bi] == ci)
                        if temp == 0: temp = 1
                        class_num_count.append(temp)
                    bs_class_num_count.append(class_num_count)
                bs_class_num_count = torch.tensor(bs_class_num_count)
                bs_class_num_count = bs_class_num_count.reshape((args.batch_size, args.classes, 1))
                bs_class_num_count = bs_class_num_count.expand([args.batch_size, args.classes, 512 * 512])
                bs_class_num_count = bs_class_num_count.long().to(device)

                out, fea = model(x_train)
                if fea.shape[-1] != 512:
                    fea = F.interpolate(fea, size=(512, 512), mode="bilinear", align_corners=False)
                rand_aug = TSRM(args.channel, args.channel, args.kernel_size_list,
                                args.L_max, args.gamma_std, args.beta_std, True).to(device)
                rand_aug.randomize()
                # shuffle auxiliary domain samples
                idx = torch.randperm(x_train.shape[0])
                a_x = x_train[idx]
                a_y = y_train[idx]
                a_x = rand_aug(a_x).to(device)
                aug_out, aug_fea = model(a_x)
                if aug_fea.shape[-1] != 512:
                    aug_fea = F.interpolate(aug_fea, size=(512, 512), mode="bilinear", align_corners=False)

                mask = F.one_hot(y_train, args.classes)
                mask = mask.reshape([args.batch_size, args.classes, 512 * 512])
                fea = fea.reshape([args.batch_size, 512 * 512, -1])

                mask, fea = mask.float().to(device), fea.float().to(device)
                mask = mask / bs_class_num_count
                aug_mask = mask[idx]

                anchor = torch.bmm(mask, fea)
                aug_fea = aug_fea.reshape([args.batch_size, 512 * 512, -1])
                aug_fea = aug_fea.float().to(device)
                positive = torch.bmm(aug_mask, aug_fea)
                negative = torch.ones(positive.shape)
                negative = negative.to(device)
                anchor = anchor.reshape([args.classes * args.batch_size, -1])
                positive = positive.reshape([args.classes * args.batch_size, -1])
                negative = negative.reshape([args.classes * args.batch_size, -1])

                loss1, loss1_num = 0, 0
                for ci in range(args.classes):
                    for cj in range(args.classes):
                        if ci == cj:
                            continue
                        else:
                            negative[ci] *= positive[cj]
                            anchor, positive, negative = F.normalize(anchor, dim=1), F.normalize(positive,
                                                                                                 dim=1), F.normalize(
                                negative, dim=1)
                            loss1 += triplet_loss(anchor, positive, negative, alpha=args.xi)
                            loss1_num += 1

                loss1 /= loss1_num
                loss2 = (criterion(out, y_train) + criterion(aug_out, a_y)) / 2
                loss = loss1 + loss2

                seg_loss += loss2.item() * len(y_train)
                align_loss += loss1 * args.classes * args.batch_size
                train_loss += loss.item() * len(y_train)

                loss.backward()
                optimizer.step()
                train_num += len(y_train)

                model.eval()
                x_val = x_val.float().to(device)
                y_val = y_val.long().to(device)
                out, fea = model(x_val)
                loss = criterion(out, y_val)
                val_loss += loss.item() * len(y_val)
                val_num += len(y_val)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
        print('Seg loss: {:.4f}, Align loss: {:.4f}, Train loss: {:.4f}, Val Loss: {:.4f}'.format(
            seg_loss / train_num, align_loss / train_num, train_loss_all[-1], val_loss_all[-1]))

    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all
        }
    )
    model.load_state_dict(best_model_wts)
    return model, train_process


model, train_process = train_model(
    model=model, criterion=criterion, optimizer=optimizer, data_loader=data_loader, num_epochs=args.epoch
)

torch.save(model, args.save_weight_path)
plt.figure(figsize=(10, 6))
plt.plot(train_process.epoch, train_process.train_loss_all, label="Tran loss")
plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.savefig(args.save_png_path)
