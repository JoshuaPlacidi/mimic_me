from torch.utils.data import Dataset, DataLoader
from message_parser import WhatsAppParser

import random

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
		
def create_dataloaders(username: str, data_folder: str, batch_size: int = 32, split_ratio: float = 0.8):
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
	# create parser object and use it to generate datapoints
	parser = WhatsAppParser(username, debug=False)
	datapoints = parser.parse_folder(data_folder)

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