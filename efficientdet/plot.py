import matplotlib.pyplot as plt
from train import*
from train import train_epoch

def plot():
        plots = {}
        plots = train_epoch(epoch, model, loader, optimizer, args)
        print(plots)
        plt.plot(plots[0], range(1, len(epoch)))
        plt.show()
