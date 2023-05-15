import os
import warnings
# warnings.filterwarnings("ignore")

from options import TrainOptions
from utils.hp_search import HPSearcher
from utils.train_student import Trainer

import random
import numpy as np
import torch


def main():
    args = TrainOptions().initialize()

    if args.reproduce:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.hp_search:
        searcher = HPSearcher(args)
        searcher.hp_search()
    else:
        trainer = Trainer(args)
        trainer.train()


if __name__ == "__main__":
    main()
