import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default=0, type=int)  # Number of statistical runs
parser.add_argument("--name", type=str)
parser.add_argument("--result_directory", nargs='+')
parser.add_argument("--mode", type=str, default="ret")
args = parser.parse_args()

steps = np.load(f'results/{args.result_directory[0]}/run_0/evaluations.npz')['timesteps']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for algorithm in args.result_directory:
    returns = np.average(np.array([np.load(f'results/{algorithm}/run_{run}/evaluations.npz')['results'][:steps.size] for run in range(args.runs)]), axis=2)
    returns_len = np.average(np.array([np.load(f'results/{algorithm}/run_{run}/evaluations.npz')['ep_lengths'][:steps.size] for run in range(args.runs)]), axis=2)

    if args.mode == "ret":
        ax.plot(steps, np.average(returns, axis=0), "-", linewidth=1, label=f'{algorithm}')
        ax.fill_between(steps, np.min(returns, axis=0), np.max(returns, axis=0), alpha=0.2)
    elif args.mode == "len":
        ax.plot(steps, np.average(returns_len, axis=0), "-", linewidth=1, label=f'{algorithm}')
        ax.fill_between(steps, np.min(returns_len, axis=0), np.max(returns_len, axis=0), alpha=0.2)

    print(f"{algorithm} --- max return: {returns.max()}, --- on run: {returns.max(axis=1).argmax()}")


if args.mode == "ret":
    ax.set_title(f'SB3 on Gymnasium/MuJoCo/{args.name}, for {args.runs} Runs, episodic return')
    ax.set_ylabel("Episode Return")
elif args.mode == "len":
    ax.set_title(f'SB3 on Gymnasium/MuJoCo/{args.name}, for {args.runs} Runs, episodic lenght')
    ax.set_ylabel("Episode Steps")

ax.legend(loc="upper left")

fig.set_figwidth(16)
fig.set_figheight(9)

for file_extenion in ["png", "pdf"]:
    fig.savefig(f"figures/{args.name}_{args.mode}.{file_extenion}", bbox_inches="tight")
