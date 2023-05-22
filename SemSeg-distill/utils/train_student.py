from networks.kd_model import load_T_model
import optuna
import sys
import torchvision
import torch.distributed as dist
import torch
import numpy as np
import random
from utils.criterion import CriterionDSN, CriterionCWD
from networks.pspnet_old import Res_pspnet as PSPNet_Student
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck, TransferConv
from utils.evaluate import evaluate_main
from networks.kd_model import NetModel, load_S_model
from dataset.datasets import CSTrainValSet, VOCDataSet, PascalVOCSearchDataset
from options import TrainOptions
import os
import warnings
import logging
from alive_progress import alive_it
from alive_progress import config_handler
config_handler.set_global(length=15, force_tty=True, file=sys.stderr)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


class Trainer:

    def __init__(self, args, trial=None):

        self.args = args
        self.trial = trial

        if should_distribute():
            logging.info(f"Using distributed PyTorch with {args.backend} backend")
            dist.init_process_group(backend=args.backend)

        self.get_data()

        self.get_teachers()

        trainable_list = torch.nn.ModuleList([])

        # self.model = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
        self.model = PSPNet_Student(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes)
        self.model = load_S_model(args, self.model)

        self.model.cuda()

        if is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.rank = dist.get_rank()
            self.is_distributed = True
        else:
            self.rank = 0
            self.is_distributed = False

        trainable_list.append(self.model)

        self.criterion_dsn = CriterionDSN().cuda()
        self.criterion_cwd = CriterionCWD(args.norm_type, args.divergence, args.temperature).cuda()

        self.G_solver = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, trainable_list.parameters()),
                                          'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        self.G_loss = 0.0

        if args.verbose:
            for key, val in args._get_kwargs():
                logging.info(key + ' : ' + str(val))

    def get_teachers(self):
        self.teachers = []
        for d in ['L', 'M', 'S']:
            t = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes=self.args.num_classes, pretrained=False).eval().cuda()
            t.load_state_dict(torch.load(f'{self.args.T_path}/{d}.pth'))
            self.teachers.append(t)

    def get_data(self):
        self.h, self.w = map(int, self.args.input_size.split(','))

        if self.args.dataset == 'cityscapes':
            train_dataset = CSTrainValSet(self.args.data_dir, self.args.data_list,
                                          max_iters=self.args.num_steps * self.args.batch_size, crop_size=(self.h, self.w),
                                          scale=self.args.random_scale, mirror=self.args.random_mirror, domain=None)
            val_dataset = CSTrainValSet(self.args.data_dir, self.args.data_listval, crop_size=(1024, 2048),
                                        scale=False, mirror=False, domain=None)

        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True, num_workers=24, pin_memory=True)
        self.valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=24, pin_memory=True)

    def train(self):
        for step, data in alive_it(enumerate(self.trainloader, self.args.last_step), total=self.args.num_steps):
            self.adjust_learning_rate(self.args.lr_g, self.G_solver, step)
            self.set_input(data)
            self.optimize_parameters(step)
            self.print_info(step)
            if (((step + 1) >= self.args.save_ckpt_start)
                    and ((step + 1 - self.args.save_ckpt_start) % self.args.save_ckpt_every == 0)) \
                    or (step + 1 == self.args.num_steps):
                self.save_ckpt(step)
                mean_IU, IU_array = evaluate_main(self.model, self.valloader, '512,512', self.args.num_classes,
                                                  whole=True, recurrence=1, split='val', save_out=self.args.save_out)
                logging.info('mean_IU: {:.4f}  IU_array: \n{}'.format(mean_IU, IU_array))
        return mean_IU

    def set_input(self, data):
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1.0 - float(iter) / (max_iter))**(power))

    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, int(args.num_steps), args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def segmentation_forward(self):
        self.preds_S = self.model.train()(self.images)
        # self.preds_S = [s.cuda() for s in self.preds_S]
        with torch.no_grad():
            self.preds_T = torch.mean(torch.stack([t.eval()(self.images)[0] for t in self.teachers]), dim=0)
            self.preds_T = torch.nn.functional.interpolate(self.preds_T, (65, 65), mode='bilinear', align_corners=True)

    def segmentation_backward(self):
        g_loss = 0
        cce = self.criterion_dsn(self.preds_S, self.labels)
        kd = self.args.lambda_cwd * self.criterion_cwd(self.preds_S[0], self.preds_T)
        self.mc_G_loss = cce.item()
        self.kd_G_loss = kd.item()
        g_loss = g_loss + cce + kd
        g_loss.backward()
        self.G_loss = g_loss.item()

    def print_info(self, step):
        if not step % self.args.log_freq:
            out = f'lr:{self.G_solver.param_groups[-1]["lr"]:.4f} loss:{self.G_loss:.4f} |'
            out += f' ce:{self.mc_G_loss:.4f}'
            out += f' kd:{self.kd_G_loss:.4f}'
            print(out)

    def save_ckpt(self, step):
        if self.rank == 0:
            args = self.args
            logging.info('saving ckpt: ' + args.save_path + '/' + args.dataset + '_' + str(step) + '_T.pth')
            torch.save(self.model.state_dict(), args.save_path + '/' + args.dataset + '_' + str(step) + '_T.pth')

    def optimize_parameters(self, step=0):
        self.step = int(step)
        self.segmentation_forward()
        self.G_solver.zero_grad()
        self.segmentation_backward()
        self.G_solver.step()


def main():
    args = TrainOptions().initialize()

    if args.reproduce:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
