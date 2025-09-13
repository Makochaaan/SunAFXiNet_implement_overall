import matplotlib.pyplot as plt
losses = []

with open("../slurm-22303.out", "r") as f:
    for line in f.readlines():
        if line.startswith("Stage 2"):
            print(line)
            losses.append(float(line.split(" ")[-1]))

def plot_losses(losses, save_path=None):
    """
    losses: list or array of loss values (e.g., per epoch)
    save_path: if specified, saves the plot to this path
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()

plot_losses(losses[1:], save_path='loss_curve.png')
