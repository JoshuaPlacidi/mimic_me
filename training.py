import torch
from tqdm import tqdm
import datetime
import os

def train(
	model: torch.nn.Module,
	train_dataloader: torch.utils.data.Dataset,
	test_dataloader: torch.utils.data.Dataset,
	num_epochs: int = 10,
	eval_steps: int = None,
	optimizer = torch.optim.AdamW,
	initial_learning_rate: float = 1e-5,
	train_layers: int = 3,
	log_filepath: str = 'training.log',
	log_test_prompt: str = "Hello, tell me about yourself"
	):

	'''
	Description:
		Train a model using a train dataloader and test dataloader

	Params:
		- model: PyTorch model to train
		- train_dataloader: PyTorch dataloader storing training data
		- test_dataloader: PyTorch dataloader storing data to evaluate model performance on
		- num_epochs: number of times the full training set gets passed through the model
		- eval_steps: how frequently to test the performance of the model
		- optimizer: PyTorch optim object used to adjust model paramaters during training
		- criterion: PyTorch loss function, quantifies the performance of the model
		- initial_learning_rate: The initial size of parameter adjustments, lr gets reduced during training by a scheduler
		- device: which hardware component (device) to run model training on, typically "cpu" or "cuda"
	'''
	# the total number of layers in the model
	NUM_LAYERS = 11

	# create training folder
	folder_name = datetime.datetime.now().strftime('%d_%m_%Y__%H_%M')
	folder_path = os.path.join('training_outputs', folder_name)
	os.mkdir(folder_path)

	# tqdm object to create progress bar for epoch count
	epoch_bar = tqdm(range(num_epochs), desc='Epoch Progress Bar', position=0)

	# we only want to train the deocder, so disable gradient calculations for the encode
	for name, child in model.model.model.named_children():
		for name, param in child.named_parameters():

			if name.startswith('layers.'):

				layer_num_start_index = len('layers.')
				layer_num_end_index = layer_num_start_index + name[layer_num_start_index:].find('.')

				layer_num = int(name[layer_num_start_index:layer_num_end_index])

				# this will be true for layers which we do not wish to train, we only train the last train_layers number of layers
				if layer_num <= (NUM_LAYERS - train_layers):
					param.requires_grad = False
				else:
					continue

	# initialise the optimizer with the model parameters we would like to be adjusted during training
	optimizer = optimizer(
		params=filter(lambda p: p.requires_grad, model.parameters()),
		lr=initial_learning_rate,
	)
	
	# a learning rate scheduler that automatically reduces the lr when loss on the evaluation set fails to reduce over a peroid of time
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer = optimizer,
		mode = 'min',
		factor = 0.1,
		patience = 2,
		threshold = 0.005,
		cooldown = 10,
	)

	# if eval_steps has not be specified than set evaluation to run 4 times per epoch
	if eval_steps == None:
		eval_steps = len(train_dataloader) // 4
		eval_steps = max(1, eval_steps)

	# lists for storing training stats
	train_loss_list = []
	
	epochs = []
	all_train_loss = []
	all_eval_loss = []
	all_lr = []

	# used to store the best evaluation performance seen so far, initialised to large dummy value
	best_eval_loss = 1e10
	eval_loss = 0

	for epoch in epoch_bar:

		for iteration, batch in enumerate(tqdm(train_dataloader, desc='Batch Progress Bar', position=1, leave=False)):
			
			# read each batch and perform a forward pass
			output = train_forward(model=model, batch=batch)
			train_loss = output.loss

			# backward pass
			train_loss.backward()
			optimizer.step()

			# store the training loss
			train_loss_list.append(train_loss.item())

			if iteration % eval_steps == 0 and iteration > 0:
				# if eval_steps has been reached then perform an evaluation
				eval_loss = evaluate(model, test_dataloader)

				# calculate the train and validation loss for outputing/logging
				log_train_loss = round(sum(train_loss_list) / len(train_loss_list), 3)
				log_val_loss = round(eval_loss, 3)

				# get the current learning rate
				current_lr = scheduler.optimizer.param_groups[0]['lr']

				# update the progress bar
				epoch_bar.set_description(
					'Epoch Progress Bar: lr {0} | Train Loss {1} | Eval Loss {2}'.format(
						current_lr,
						log_train_loss,
						log_val_loss,
					)
				)

				train_loss_list = []
				model_saved = False

				# attempt to generate test prompt response
				model_output = model.inference(log_test_prompt)

				# if this is the lowest eval loss seen so far then save the model parameters
				if eval_loss <= best_eval_loss:
					best_eval_loss = eval_loss

					torch.save(model.state_dict(), os.path.join(folder_path, 'model.pt'))
					model_saved = True

				# store sample outputs in a text file to see model development over time
				with open(os.path.join(folder_path, log_filepath), "a+") as log_file:
						log_file.write(",".join([
								str(epoch),
								str(iteration),
								str(log_train_loss),
								str(log_val_loss),
								str(current_lr),
								str(model_saved),
								'"' + model_output + '"',
							])
							+ "\n" # add new line character to the end of the string
						)

				# if our learning rate drops below 1e-9 then end training, as not learning
				# will happend with a learning rate this small
				if current_lr < 1e-9:
					return

		# update the scheduler using the latest training loss value			
		scheduler.step(train_loss)

def train_forward(model, batch):
	'''
	Description:
		Perform a forward pass of a batch through the model

	Params:
		- batch: tuple storing sets of prompt answer pairs

	Returns:
		- loss: the models loss on the passed batch
	'''
	prompt_str, answer_str = batch
	
	# tokenize the prompt and answer
	prompt_tokens, prompt_mask = model.encode(prompt_str)
	answer_tokens, _ = model.encode(answer_str)

	# pass through model
	output = model(prompt_tokens=prompt_tokens.to(model.device), prompt_mask=prompt_mask.to(model.device), labels=answer_tokens.to(model.device))

	return output

def evaluate(model, dataloader):
	'''
	Description:
		Evaluate a models performance on a set of data

	Params:
		- dataloader: PyTorch dataloader object storing the data to evaluate on

	Returns:
		- eval_loss: the mean total loss of the model on all the data in the dataloader
	'''

	# dont generate gradients
	with torch.no_grad():

		eval_loss = 0
	
		for batch in tqdm(dataloader, desc='Running Evaluation', leave=False):
			# loop through dataloader and calculate loss
			eval_loss += train_forward(model, batch).loss.item()

		# calculalate mean
		eval_loss /= max(len(dataloader),1)

	return eval_loss