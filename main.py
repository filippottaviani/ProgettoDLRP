import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("C:\\", "Users", "FilippoOttaviani", "PycharmProjects", "ProgettoDLRP", "dataset"))

    parser.add_argument("--is_train", type=bool, default=True) # --is_train=False per effettuare il test
    parser.add_argument("--ckpt_file", type=str, default="depth_10.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        solver.globalNetFit()
        solver.adversarialFit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
