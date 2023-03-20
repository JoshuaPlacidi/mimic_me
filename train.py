import sys
import torch
from dataset import create_dataloaders
from model import ChatModel
from training import train
import argparse
import pickle


if __name__ == '__main__':
    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for running training using your generated datapoints')

    # set arguments
    parser.add_argument('-d', '--datapoints_filepath', type=str, required=True,
                        help='The path to the pickle file containing the generate datapoints')
    parser.add_argument('-c', '--cuda_id', type=int, default=None,
                        help='If using a GPU then specify which cuda device you would like to use')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='The size of data batches to use during training, smaller batch sizes is less computationally intensive')


    # extract arguments
    args = parser.parse_args()
    datapoints_filepath = args.datapoints_filepath
    cuda_id = args.cuda_id
    batch_size = args.batch_size

    # format the training device, uses cuda is specified otherwise trains using cpu
    device = "cuda:{0}".format(cuda_id) if cuda_id != None else 'cpu'

    # try to load datapoints
    with open(datapoints_filepath, 'rb') as file:
        datapoints = pickle.load(file)

    # create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(datapoints, batch_size=batch_size)

    model = ChatModel(device=device)

    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=40)

    print('\nTraining complete, model state saved to model.pt')
