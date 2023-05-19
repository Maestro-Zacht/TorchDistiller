import os
import argparse
import logging
import time
import torch.distributed as dist


# Dataset
DATASET = 'cityscapes'
NUM_CLASSES = 19 if DATASET == 'cityscapes' else 21
DATA_DIRECTORY = '/mnt/dataset/CIRKD/data/Cityscapes/'

if DATASET == 'cityscapes':
    DATA_LIST_TRAIN_PATH = './dataset/list/cityscapes/train.lst'
    DATA_LIST_VAL_PATH = './dataset/list/cityscapes/val.lst'
    DATA_LIST_TEST_PATH = './dataset/list/cityscapes/test.lst'
    T_CKPT = '/mnt/dataset/CWD/ckpt/teacher_city.pth'  # '../../CWD/ckpt/resnet101.pth' #
else:
    DATA_LIST_TRAIN_PATH = '/mnt/dataset/CIRKD/data/PascalVOC/ImageSets/Segmentation/train.txt'
    DATA_LIST_VAL_PATH = '/mnt/dataset/CIRKD/data/PascalVOC/ImageSets/Segmentation/val.txt'
    DATA_LIST_TEST_PATH = None
    T_CKPT = './ckpt/train_epoch_50.pth'

IGNORE_LABEL = 255
INPUT_SIZE = '512,512'

# Weights

S_CKPT = '/mnt/dataset/CWD/ckpt/resnet18-imagenet.pth'
T_PATH = './ckpt/teachers'

# Training HPs
BATCH_SIZE = 8
NUM_STEPS = 40000
MOMENTUM = 0.9
POWER = 0.9
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0001

# Checkpoint
SAVE_CKPT_START = 5000
SAVE_CKPT_EVERY = 5000

# Reproducibility
SEED = 0

# HP Search
N_TRIALS = 36
HP_SEARCH_NAME = 'dkd'
HP_SEARCH_DIR = 'trials'


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='train')

        # Distributed
        if dist.is_available():
            parser.add_argument("--backend", type=str, help="Distributed backend",
                                choices=[dist.Backend.GLOO,
                                         dist.Backend.NCCL, dist.Backend.MPI],
                                default=dist.Backend.NCCL)

        # GPU
        parser.add_argument("--gpu", default='0', help="Choose gpu device")

        # Dataset
        parser.add_argument('--dataset', type=str, default=DATASET, help='The name of the dataset')
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='Number of classes to predict')
        parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY, help="Path to the dataset directory")
        parser.add_argument("--data_list", type=str, default=DATA_LIST_TRAIN_PATH, help="Path to training file listing")
        parser.add_argument("--data_listval", type=str, default=DATA_LIST_VAL_PATH, help="Path to the validation file listing")
        parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore")
        parser.add_argument("--input_size", type=str, default=INPUT_SIZE, help="Comma-separated string (h,w) for input shape")
        parser.add_argument("--random_mirror", action="store_true", help="Randomly mirror the inputs during the training")
        parser.add_argument("--random_scale", action="store_true", help="Randomly scale the inputs during the training")
        parser.add_argument("--domain", default='None')

        # Weights
        parser.add_argument('--T_ckpt_path', type=str, default=T_CKPT, help='teacher ckpt path')
        parser.add_argument('--T_path', type=str, default=T_PATH, help='teacher ckpt path')
        parser.add_argument('--S_resume', type=str2bool, default='False', help='is or not use student ckpt')
        parser.add_argument('--S_ckpt_path', type=str, default='', help='student ckpt path')
        parser.add_argument('--D_resume', type=str2bool, default='False', help='is or not use discriminator ckpt')
        parser.add_argument('--D_ckpt_path', type=str, default='', help='discriminator ckpt path')
        parser.add_argument("--is_student_load_imgnet", type=str2bool, default='True', help="is student load imgnet")
        parser.add_argument("--student_pretrain_model_imgnet", type=str, default=S_CKPT, help="student pretrain model on imgnet")

        # Training HPs
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of images to feed in one step")
        parser.add_argument("--num_steps", type=int, default=NUM_STEPS, help="Number of training steps")
        parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser")
        parser.add_argument("--power", type=float, default=POWER, help="Decay parameter to compute the learning rate")
        parser.add_argument("--lr_g", type=float, default=LEARNING_RATE, help="learning rate for G")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss")
        parser.add_argument("--last_step", type=int, default=0, help="last train step (to start from checkpoint)")

        # Loss
        # CE
        parser.add_argument("--ce", type=str2bool, default='True')
        # KD
        parser.add_argument("--kd", type=str2bool, default='False')
        parser.add_argument("--lambda_kd", type=float, default=1.0, help="lambda_kd")
        # EKD
        parser.add_argument("--ekd", type=str2bool, default='False')
        parser.add_argument("--lambda_ekd", type=float, default=1.0, help="lambda_ekd")
        # CWD
        parser.add_argument("--cwd", type=str2bool, default='True')
        parser.add_argument("--cwd_feat", type=str2bool, default='False')
        parser.add_argument("--temperature", type=float, default=4.0, help="normalize temperature")
        parser.add_argument("--lambda_cwd", type=float, default=3.0, help="lambda_kd")
        parser.add_argument("--norm_type", type=str, default='channel', help="kd normalize setting")
        parser.add_argument("--divergence", type=str, default='kl', help="kd divergence setting")
        # ADV
        parser.add_argument("--adv", type=str2bool, default='False')
        parser.add_argument("--lambda_adv", type=float, default=0.001, help="lambda_adv")
        parser.add_argument("--preprocess_GAN_mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
        parser.add_argument("--adv_loss_type", type=str, default='wgan-gp', help="adversarial loss setting")
        parser.add_argument("--imsize_for_adv", type=int, default=65, help="imsize for addv")
        parser.add_argument("--adv_conv_dim", type=int, default=64, help="conv dim in adv")
        parser.add_argument("--lambda_gp", type=float, default=10.0, help="lambda_gp")
        parser.add_argument("--lambda_d", type=float, default=0.1, help="lambda_d")
        parser.add_argument("--lr_d", type=float, default=4e-4, help="learning rate for D")
        # IFV
        parser.add_argument("--ifv", type=str2bool, default='False')
        parser.add_argument('--lambda_ifv', type=float, default=200.0, help='lambda_ifv')
        # AKD
        parser.add_argument("--akd", type=str2bool, default='False')
        parser.add_argument('--lambda_akd', type=float, default=1.0, help='lambda_akd')
        parser.add_argument('--k', type=float, default=0.5, help='k')
        # SRRL
        parser.add_argument("--srrl", type=str2bool, default='False')
        parser.add_argument('--lambda_srrl_reg', type=float, default=1.0, help='lambda_srrl')
        parser.add_argument('--lambda_srrl_feat', type=float, default=1.0, help='lambda_srrl')
        parser.add_argument('--reg_loss', type=str, default='mse', help='srrl regression loss')
        parser.add_argument('--srrl_layer', type=str, default='last', help='srrl feature layer')
        # MGD
        parser.add_argument("--mgd", type=str2bool, default='True')
        parser.add_argument('--lambda_mgd', type=float, default=0.001, help='lambda_mgd')
        parser.add_argument('--alpha_mgd', type=float, default=0.25, help='alpha_mgd')
        parser.add_argument('--mgd_layer', type=str, default='back', help='mgd feature layer')
        parser.add_argument('--mgd_mask', type=str, default='channel', help='mgd feature mask')
        # DKD
        parser.add_argument("--dkd", type=str2bool, default='True')
        parser.add_argument('--lambda_dkd', type=float, default=0.1, help='lambda_dkd')
        parser.add_argument('--alpha_dkd', type=float, default=1.0, help='alpha_dkd')
        parser.add_argument('--beta_dkd', type=float, default=8.0, help='beta_dkd')
        parser.add_argument("--temp_dkd", type=float, default=4.0, help="temp_dkd")
        parser.add_argument('--norm_dkd', type=str, default='channel', help='dkd feature norm')
        parser.add_argument('--warmup_dkd', type=float, default=400, help='dkd warmup')

        # Checkpoint
        parser.add_argument("--save_name", type=str, default='exp')
        parser.add_argument("--save_dir", type=str, default='ckpt', help="Where to save models")
        parser.add_argument("--save_ckpt_start", type=int, default=SAVE_CKPT_START)
        parser.add_argument("--save_ckpt_every", type=int, default=SAVE_CKPT_EVERY)
        parser.add_argument("--log_freq", type=int, default=200, help="Number of training steps")
        parser.add_argument("--save_out", action="store_true", help="Use a fixed seed")

        # Reproducibility
        parser.add_argument("--reproduce", action="store_true", help="Use a fixed seed")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--verbose", action="store_true", help="Verbose")

        # HP Search
        parser.add_argument("--hp_search", action="store_true", help="Run hyperparameter search")
        parser.add_argument("--n_trials", type=int, default=N_TRIALS, help="Number of trials")
        parser.add_argument("--hp_search_name", default=HP_SEARCH_NAME, type=str, help="Hyperparameter search name")
        parser.add_argument("--hp_search_path", default=HP_SEARCH_DIR, type=str, help="Hyperparameter search directory")

        try:
            args = parser.parse_args()
        except:
            args = parser.parse_args('')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_path = args.save_dir + '/' + args.save_name
        log_init(args.save_path, args.dataset)

        return args


class ValOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='Val')
        parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
        parser.add_argument("--data_list", type=str, default=DATA_LIST_VAL_PATH)
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
        parser.add_argument("--restore_from", type=str, default='')
        parser.add_argument("--gpu", default='0')

        args = parser.parse_args()

        for key, val in args._get_kwargs():
            print(key + ' : ' + str(val))

        return args


class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='Test')
        parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
        parser.add_argument("--data_list", type=str, default=DATA_LIST_TEST_PATH)
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
        parser.add_argument("--restore_from", type=str, default='')
        parser.add_argument("--gpu", default='0')

        args = parser.parse_args()

        for key, val in args._get_kwargs():
            print(key + ' : ' + str(val))

        return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def log_init(log_dir, name='log'):
    time_cur = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_dir + '/' + name + '_' + str(time_cur) + '.log',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
