import sys
import torch
from dataset import create_dataloaders
from model import ChatModel
from training import train
import argparse
import pickle

datapoints_filepath = 'datapoints.pkl'

# try to load datapoints
with open(datapoints_filepath, 'rb') as file:
    datapoints = pickle.load(file)

# create dataloaders
train_dataloader, test_dataloader = create_dataloaders(datapoints, batch_size=5)


for batch in train_dataloader:
    for datapoint in batch:
        print(datapoint)
        print()
    exit()