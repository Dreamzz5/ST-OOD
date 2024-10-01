import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.models.astgcn import ASTGCN
from src.engines.astgcn_engine import ASTGCN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--nb_block", type=int, default=2)
    parser.add_argument("--nb_chev_filter", type=int, default=64)
    parser.add_argument("--nb_time_filter", type=int, default=64)
    parser.add_argument("--time_stride", type=int, default=1)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()
    args.model_name = __file__.split("/")[-2]
    folder_name = "{}_{}".format(args.dataset, args.years)
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info("Adj path: " + adj_path)

    adj = load_adj_from_numpy(adj_path)
    np.fill_diagonal(adj, 0)

    # adj = np.zeros((node_num, node_num), dtype=np.float32)
    # for n in range(node_num):
    #     idx = np.nonzero(adj_mx[n])[0]
    #     adj[n, idx] = 1

    L_tilde = normalize_adj_mx(adj, "scalap")[0]
    cheb_poly = [
        torch.from_numpy(i).type(torch.FloatTensor).to(device)
        for i in calculate_cheb_poly(L_tilde, args.order)
    ]

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = ASTGCN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=args.device,
        cheb_poly=cheb_poly,
        order=args.order,
        nb_block=args.nb_block,
        nb_chev_filter=args.nb_chev_filter,
        nb_time_filter=args.nb_time_filter,
        time_stride=args.time_stride,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    engine = ASTGCN_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate("test")
        engine.evaluate("shift")


if __name__ == "__main__":
    main()
