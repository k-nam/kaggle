import matplotlib.pyplot as plt
import tensorflow_docs.plots


def plot_model_history(model_history, metric, ylim=None):
    plt.style.use('seaborn-darkgrid')
    plotter = tensorflow_docs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)
    plt.title(f'{metric.upper()}')
    if ylim is None:
        plt.ylim([0, 1])
    else:
        plt.ylim(ylim)

    # plt.savefig(f'{metric}.png')
    # plt.close()