import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

import datasets
import utils

from model_resnet import ResidualNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
import math
import time
import datetime
from tensorboardX import SummaryWriter
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [0]',
        default=0,
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='Maximum number of training epochs.',
        default=25,
        type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Batch size.',
        default=10, type=int)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='Base learning rate.',
        default=0.00006, type=float)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Dataset type.',
        default='BIWI_Pose_300W_LP',
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='Directory path for data.',
        # default='/media/omnisky/D4T/huli/work/with_mask_dockerface/',
        default='/media/omnisky/D4T/huli/work/headpose/data',
        type=str)
    parser.add_argument(
        '--filename_list',
        dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='/media/omnisky/D4T/huli/work/headpose/data/file_name_biwi_300w_lp_no_mask20210212.txt',
        # Filename_biwi_300w_lp 有口罩和没有口罩，2021年3月12日17:16:20
        # default='/media/omnisky/D4T/huli/work/headpose/data/Filename_biwi_300w_lp.txt',
        # default='/media/omnisky/D4T/huli/work/with_mask_dockerface/img_name.txt',
        type=str)
    parser.add_argument(
        '--output_string',
        dest='output_string',
        help='String appended to output snapshots.',
        default='',
        type=str)
    parser.add_argument(
        '--alpha',
        dest='alpha',
        help='Regression loss coefficient.',
        default=2,
        type=float)
    parser.add_argument(
        '--snapshot',
        dest='snapshot',
        help='Path of model snapshot.',
        # default='/media/omnisky/D4T/huli/work/headpose/output/no_mask_01_gray_mix_biwi_300w_lp/gray_biwi_300W_LP_squire_epoch_5.pkl',
        default='',
        type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def compute_loss(args, axis, labels, cont_labels, preds, cls_criterion,
                 reg_criterion, idx_tensor, gpu):

    if axis == "yaw":
        dim = 0
    elif axis == "pitch":
        dim = 1
    elif axis == "roll":
        dim = 2
    else:
        raise IndexError("{} is not in ['yaw', 'pitch', 'roll']".format(axis))

    label = Variable(labels[:, dim]).cuda(gpu)
    label_cont = Variable(cont_labels[:, dim]).cuda(gpu)

    loss_cls = criterion(preds, label)
    predicted = softmax(preds)
    predicted = torch.sum(predicted * idx_tensor, 1) * 3 - 99
    loss_reg = reg_criterion(predicted, label_cont)
    loss = loss_cls + alpha * loss_reg

    return loss


def compute_error(axis, cont_labels, preds, idx_tensor):

    if axis == "yaw":
        dim = 0
    elif axis == "pitch":
        dim = 1
    elif axis == "roll":
        dim = 2
    else:
        raise IndexError("{} is not in ['yaw', 'pitch', 'roll']".format(axis))

    label_cont = cont_labels[:, dim].float()
    predictions = utils.softmax_temperature(preds.data, 1)
    predictions = torch.sum(predictions * idx_tensor, 1).cpu() * 3 - 99
    error = torch.sum(torch.abs(predictions - label_cont))

    return error


def train(args, train_loader, model, criterion,
          reg_criterion, idx_tensor, optimizer,
          epoch, num_epochs, batch_num):

    for i, (images, labels, cont_labels, name) in enumerate(train_loader):
        images = Variable(images).cuda(gpu)

        # Forward pass
        yaw, pitch, roll = model(images)

        # losses
        loss_yaw = compute_loss(
            args, "yaw", labels, cont_labels, yaw, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_pitch = compute_loss(
            args, "pitch", labels, cont_labels, pitch, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_roll = compute_loss(
            args, "roll", labels, cont_labels, roll, criterion,
            reg_criterion, idx_tensor, gpu)

        loss_seq = [loss_yaw, loss_pitch, loss_roll]
        grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in
                    range(len(loss_seq))]
        optimizer.zero_grad()
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(('Epoch [{:d}/{:d}] Iter [{:d}/{:d}] Losses:' +
                   'Yaw {:4f}, Pitch {:4f}, Roll {:4f}').format(
                      epoch + 1, num_epochs,
                      i + 1, batch_num,
                      loss_yaw.item(), loss_pitch.item(), loss_roll.item()))


def valid(valid_loader, model, idx_tensor):

    model.eval()
    total, yaw_error, pitch_error, roll_error = 0, 0.0, 0.0, 0.0

    for i, (images, labels, cont_labels, name) in enumerate(valid_loader):

        with torch.no_grad():
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)
            # Forward pass
            yaw, pitch, roll = model(images)
            yaw_error += compute_error(
                "yaw", cont_labels, yaw, idx_tensor)
            pitch_error += compute_error(
                "pitch", cont_labels, pitch, idx_tensor)
            roll_error += compute_error(
                "roll", cont_labels, roll, idx_tensor)

    print('Valid error in degrees ' +
          str(total) +
          ' test images. Yaw: {:4f}, Pitch: {:4f}, Roll: {:4f}'.format(
              yaw_error / total, pitch_error / total, roll_error / total))


if __name__ == '__main__':
    valid_data_path = "./data/AFLW2000/filename_list.txt"
    args = parse_args()
    writer = SummaryWriter("out/train_no_mask_02_gray_mix_biwi_cosin_224")
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    gpu = args.gpu_id

    if not os.path.exists('output/no_mask_02_gray_mix_biwi_300w_lp_cosin_224'):
        os.makedirs('output/no_mask_02_gray_mix_biwi_300w_lp_cosin_224')

    # net structure
    model = ResidualNet("ImageNet", 50, 66, "CBAM")

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


    print('Loading data.')

    transformations = transforms.Compose(
        [transforms.Resize(240),
         transforms.RandomCrop(224),
         #brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05
         transforms.ColorJitter(brightness=0.85,contrast=0.5,saturation=0.5,hue=0.05),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.392,0.392,0.392],
             std=[0.254, 0.254, 0.254])])
    valid_transform = transforms.Compose(
        [transforms.Scale(224),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.392,0.392,0.392],
             std=[0.254, 0.254, 0.254])])
    #减小一半输入
    # transformations = transforms.Compose(
    #     [transforms.Resize(114),
    #      transforms.RandomCrop(112),
    #      #brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05
    #      transforms.ColorJitter(brightness=0.85,contrast=0.5,saturation=0.5,hue=0.05),
    #      transforms.ToTensor(),
    #      transforms.Normalize(
    #          mean=[0.392,0.392,0.392],
    #          std=[0.254, 0.254, 0.254])])
    # valid_transform = transforms.Compose(
    #     [transforms.Scale(114),
    #      transforms.CenterCrop(112),
    #      transforms.ToTensor(),
    #      transforms.Normalize(
    #          mean=[0.392,0.392,0.392],
    #          std=[0.254, 0.254, 0.254])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(
            args.data_dir,
            args.filename_list,
            transformations)
    elif args.dataset == 'BIWI_Pose_300W_LP':
            pose_dataset = datasets.BIWI_Pose_300W_LP(
                args.data_dir,
                args.filename_list,
                transformations
            )
    else:
        print('Error: not a valid dataset name')
        sys.exit()
    val_data_dir = "data/AFLW2000/"
    val_filename_list = "data/AFLW2000/filename_list.txt"
    valid_pose_dataset = datasets.AFLW2000(
            val_data_dir,
            val_filename_list,
            valid_transform)
    valid_dataset = DataLoader(dataset=valid_pose_dataset,batch_size=16,num_workers=8)
    #get len of valid_dataset
    vaild_total = len(valid_pose_dataset)
    # train_size = int(0.9 * len(pose_dataset))
    # valid_size = len(pose_dataset) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(
    #     pose_dataset, [train_size, valid_size])
    epoch_size = math.ceil(len(pose_dataset)/batch_size)
    max_iter = num_epochs*epoch_size

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8)

    # valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=2)

    model.cuda(gpu)
    #model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    # reg_criterion = nn.SmoothL1Loss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)



    optimizer = torch.optim.Adam(
        [{'params': get_ignored_params(model), 'lr': 0},
         {'params': get_non_ignored_params(model), 'lr': args.lr},
         {'params': get_fc_params(model), 'lr': args.lr * 5}],
        lr=args.lr)
    # optimizer = torch.optim.SGD([{'params': get_ignored_params(model), 'lr': 0},
    #      {'params': get_non_ignored_params(model), 'lr': args.lr},
    #      {'params': get_fc_params(model), 'lr': args.lr * 5}],
    #     lr=args.lr,momentum=0.9,weight_decay=0.0005)
    # SGD学习优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)


    # scheduler = MultiStepLR(optimizer, milestones=[8, 18], gamma=0.1)
    # 余弦学习率下降，加预热训练 max_epoch=30
    # warm_up_epochs = 5
    # warm_up_epoch = 35
    # warm_up_with_cosine_lr = lambda epoch: (((epoch+1)*0.00005) / warm_up_epochs) if epoch < warm_up_epochs \
    # else 0.00005 * ( math.cos((epoch - warm_up_epochs) /(warm_up_epoch - warm_up_epochs) * math.pi) + 1)

    # scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)
    #指数衰减
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=30)
    print('Ready to train network.')
    # model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()
    model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()
    # for epoch in range(num_epochs):
    cosin_copunt = 0
    epoch = 0
    for iteration in range(max_iter):
        
        
        if iteration % epoch_size == 0:
            cosin_copunt = 0
            batch_iterator = iter(DataLoader(
                                            dataset=pose_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8))
            
            if epoch>0:
                
                print('Saving checkpoint...')
                torch.save(model.state_dict(),'output/no_mask_02_gray_mix_biwi_300w_lp_cosin_224/' + args.output_string +'gray_biwi_300W_LP_squire_epoch_' + str(epoch+1) + '.pkl')
                # valid data
                val_yaw_error = .0
                val_pitch_error = .0
                val_roll_error = .0
                with torch.no_grad():
                    for i,(val_imgs,val_labels,val_const_labels,val_names) in enumerate(valid_dataset):
                        val_label_yaw = val_const_labels[:,0].float()
                        val_label_pitch = val_const_labels[:,1].float()
                        val_label_roll = val_const_labels[:,2].float()
                        val_yaw,val_pitch,val_roll = model(val_imgs.cuda())

                        # Binned predictions
                        _, yaw_bpred = torch.max(val_yaw.data, 1)
                        _, pitch_bpred = torch.max(val_pitch.data, 1)
                        _, roll_bpred = torch.max(val_roll.data, 1)

                        # Continuous predictions
                        yaw_predicted = utils.softmax_temperature(val_yaw.data, 1)
                        pitch_predicted = utils.softmax_temperature(val_pitch.data, 1)
                        roll_predicted = utils.softmax_temperature(val_roll.data, 1)

                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
                        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

                        # Mean absolute error
                        val_yaw_error += torch.sum(torch.abs(yaw_predicted - val_label_yaw))
                        val_pitch_error += torch.sum(torch.abs(pitch_predicted - val_label_pitch))
                        val_roll_error += torch.sum(torch.abs(roll_predicted - val_label_roll))
                writer.add_scalars("val_pitch_roll_yaw",{"val_err_yaw":val_yaw_error/vaild_total,'val_err_pitch':val_pitch_error/vaild_total,'val_err_roll':val_roll_error/vaild_total},epoch)
                
                
                # valid(valid_loader, model, idx_tensor)
            epoch += 1
          
        load_t0 = time.time()   

        #load data train  , name
        images, labels, cont_labels,name = next(batch_iterator)
        images.cuda()
        yaw, pitch, roll = model(images)

        loss_yaw = compute_loss(
            args, "yaw", labels, cont_labels, yaw, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_pitch = compute_loss(
            args, "pitch", labels, cont_labels, pitch, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_roll = compute_loss(
            args, "roll", labels, cont_labels, roll, criterion,
            reg_criterion, idx_tensor, gpu)

        loss_seq = [loss_yaw, loss_pitch, loss_roll]
        grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in
                    range(len(loss_seq))]
        optimizer.zero_grad()
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()
        scheduler.step(epoch+cosin_copunt/epoch_size)
        cosin_copunt += 1  
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))

        print("Epoch:{}/{} || Epochiter:{}/{} || Yaw:{:.4f},Pitch:{:.4f},Roll:{:.4f}||Batchtime:{:.4f}||ETA:{}".format(
            epoch,num_epochs,(iteration%epoch_size)+1,epoch_size,loss_yaw.item(),loss_pitch.item(),loss_roll.item(),batch_time,
            str(datetime.timedelta(seconds=eta))
        ))
        #损失为无穷大时停止运算
        
        if not math.isfinite(loss_yaw.item()+loss_pitch.item()+loss_roll.item()):
            print("Loss is {}, stopping training".format(loss_yaw.item()+loss_pitch.item()+loss_roll.item()))
            # print(losses_dict_reduced)
            sys.exit(1)
        writer.add_scalars("pitch_roll_yaw",{"loss_yaw":loss_yaw.item(),'loss_pitch':loss_pitch.item(),'loss_roll':loss_roll.item()},iteration)
        
        writer.add_scalars("lr",{"ignore":optimizer.param_groups[0]['lr'],"none_ignore":optimizer.param_groups[1]['lr'],"fc":optimizer.param_groups[2]['lr']},iteration)
        # train(args, train_loader, model, criterion, reg_criterion,
        #       idx_tensor, optimizer, epoch, num_epochs,
        #       len(train_dataset) // batch_size)
        
        

        
        # torch.save(
        #     model.state_dict(),
            # 'output/biwi/' + args.output_string +
            # '300W_LP_squire_epoch_' + str(epoch+1) + '.pkl')
