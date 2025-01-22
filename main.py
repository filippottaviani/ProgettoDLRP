import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    
    parser.add_argument("--gl_ckpt_dir", type=str, default="./checkpoint/global")
    parser.add_argument("--ref_ckpt_dir", type=str, default="./checkpoint/refinement")
    parser.add_argument("--gl_ckpt_name", type=str, default="gl_depth")
    parser.add_argument("--ref_ckpt_name", type=str, default="ref_depth")
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("C:\\", "Users", "FilippoOttaviani", "PycharmProjects", "ProgettoDLRP", "dataset"))

    parser.add_argument("--is_train", type=bool, default=True) # --is_train=False per effettuare il test
    parser.add_argument("--only_global", type=bool, default=False)  # per effettuare il test solo con la global net
    parser.add_argument("--gl_ckpt_file", type=str, default="gl_depth_100.pth")
    parser.add_argument("--ref_ckpt_file", type=str, default="ref_depth_100.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        #solver.globalnet_fit()
        #solver.refnet_pretrain()
        solver.adversarial_fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
