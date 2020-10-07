import numpy as np
import matplotlib.pyplot as plt

def main():
    for f in [ReLU, sigmoid, swish]:
        plot(f)

def plot(f):
    x = np.linspace(-5,5)
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    plt.axhline(0, color='grey', linewidth=0.3)
    plt.axvline(0, color='grey', linewidth=0.3)
    plt.plot(x,f(x),"black")
    plt.show()

def ReLU(x):
    return (x + np.abs(x)) / 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x) #in general sigmoid(b*x)


if __name__ == "__main__":
    main()