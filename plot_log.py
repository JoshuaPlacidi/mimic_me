import argparse
import matplotlib.pyplot as plt


class LogPlotter:

    def __init__(self, log_filepath: str):
        self.metrics = ['epoch', 'iteration', 'train_loss', 'val_loss', 'lr', 'model_save', 'test_prompt_response']
        self.colours = ['darkred', 'indigo', 'darkolivegreen', 'midnightblue']

        self.data = self.parse_logs(log_filepath)

    def parse_logs(self, log_filepath: str) -> dict:

        data = {}

        for metric in self.metrics:
            data[metric] = []

        with open(log_filepath, 'r') as log_file:
            for line in log_file:
                raw_log = line.split(',')

                for metric, value in zip(self.metrics, raw_log):

                    try:
                        value = float(value)
                    except:
                        pass

                    data[metric].append(value)

        return data

    def plot(self, metric_xs: str):

        num_plots = len(metric_xs)

        y_data = [e + (i / max(self.data['iteration'])) for (e, i) in zip(self.data['epoch'], self.data['iteration'])]

        for index, x_title in enumerate(metric_xs):

            plt.subplot(1, num_plots, index + 1)
            plt.plot(y_data, self.data[x_title], color=self.colours[index])
            plt.yscale("log")
            plt.title(x_title)
            plt.xlabel('epoch')
            plt.ylabel(x_title)

        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    # initialise argument parser
    parser = argparse.ArgumentParser(description='This script is for ploting log files generated during training')

    # set arguments
    parser.add_argument('-l', '--log_filepath', type=str, required=True,
                        help='The path to the pickle file containing the generate datapoints')
    # parser.add_argument('-c', '--cuda_id', type=int, default=None,
    #                     help='If using a GPU then specify which cuda device you would like to use')
    # parser.add_argument('-b', '--batch_size', type=int, default=16,
    #                     help='The size of data batches to use during training, smaller batch sizes is less computationally intensive')


    # extract arguments
    args = parser.parse_args()
    log_filepath = args.log_filepath
    
    log_plotter = LogPlotter(log_filepath)
    log_plotter.plot(metric_xs=['train_loss', 'val_loss', 'lr'])
