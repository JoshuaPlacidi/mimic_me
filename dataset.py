from torch.utils.data import Dataset, DataLoader
from message_parser import WhatsAppParser
from typing import List, Union

import random
import argparse
import pickle
import os

class ChatDataset(Dataset):
	def __init__(self, datapoints: list):
		'''
		Description:
			Dataset object storing prompt answer pairs

		Params:
			- datapoints: list of dictionary objects storing prompt and answer information for each datapoint
		'''
		self.datapoints = datapoints

	def __len__(self):
		# return number of datapoints
		return len(self.datapoints)

	def __getitem__(self, index):
		'''
		Description:
			Indexes datapoints list and returns datapoint information

		Params:
			- index: Index of desired datapoint

		Returns:
			- (prompt, answer): tuple of strings containing prompt and answer messages
		'''
		# read datapoint
		datapoint = self.datapoints[index]

		# read prompt and answer message
		context = datapoint['context']
		response = datapoint['response']

		return context, response
		
def create_dataloaders(datapoints, batch_size: int = 32, split_ratio: float = 0.8):
	'''
	Description:
		Parse a series of chat files and create dataset loaders

	Params:
		- data_folder: folder containing the chat files
		- batch_size: size of each batch to use during training/testing (smaller number is less resource intense)
		- split_ratio: the ratio of samples to use for training compared to testing

	Returns:
		- train_dataloader: PyTorch dataloader object storing training data
		- test_dataloader: PyTorch dataloader object storing testing data
	'''
	# shuffle datapoints and split into train/test
	random.shuffle(datapoints)
	split_idx = int(len(datapoints) * split_ratio)
	train_data = datapoints[:split_idx]
	test_data = datapoints[split_idx:]

	# create dataset objects
	train_dataset = ChatDataset(datapoints=train_data)
	test_dataset = ChatDataset(datapoints=test_data)

	# create dataloaders
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

	return train_dataloader, test_dataloader

if __name__ == '__main__':

	# initialise argument parser
	parser = argparse.ArgumentParser(description='This file is for generating datasets for the machine learning models')

	# set arguments
	parser.add_argument('-d','--data_dir', type=str, required=True,
						help='The path to the directory containing the individual WhatsApp chat files')
	parser.add_argument('-u', '--username', type=str, required=True,
						help='The username of the person whom you want to AI to mimic')

	# extract arguments
	args = parser.parse_args()
	data_dir = args.data_dir
	username = args.username

	# check data_directory exists
	if not os.path.isdir(data_dir):
		raise Exception('Directory "{0}" does not exist'.format(str(data_dir)))

	# generate datapoints
	parser = WhatsAppParser(username, debug=False)
	datapoints = parser.parse_folder(data_dir)

	if len(datapoints) == 0:
		raise Exception('Could not generate any datapoints. Either "{0}" does not contain chat files, or username "{1}" does not appear in any chat files'.format(str(data_dir), str(username)))

	# save datapoints using pickle
	with open('datapoints.pkl', 'wb') as file:
		pickle.dump(datapoints, file)

	print('\nDatapoints saved to "datapoints.pkl"')