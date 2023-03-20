import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(
	model: torch.nn.Module,
	train_dataloader: torch.utils.data.Dataset,
	test_dataloader: torch.utils.data.Dataset,
	num_epochs: int = 10,
	eval_steps: int = None,
	optimizer = torch.optim.AdamW,
	initial_learning_rate: float = 1e-4,
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

	# tqdm object to create progress bar for epoch count
	epoch_bar = tqdm(range(num_epochs), desc='Epoch Progress Bar', position=0)

	# we only want to train the deocder, so disable gradient calculations for the encode
	for name, child in model.model.model.named_children():
		if name == 'encoder':
			for param in child.parameters():
				param.requires_grad = False

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
		threshold = 0.005,
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
			scheduler.step(train_loss)

			# store the training loss
			train_loss_list.append(train_loss.item())

			if iteration % eval_steps == 0 and iteration > 0:
				# if eval_steps has been reached then perform an evaluation
				eval_loss = evaluate(model, test_dataloader)

				# update the progress bar
				epoch_bar.set_description(
					'Epoch Progress Bar: Iter {0}/{1} | Train Loss {2} | Eval Loss {3}'.format(
						iteration,
						len(train_dataloader),
						round(sum(train_loss_list) / len(train_loss_list), 3),
						round(eval_loss, 3),
					)
				)

				train_loss_list = []

				# if this is the lowest eval loss seen so far then save the model parameters
				if eval_loss < best_eval_loss:
					best_eval_loss = eval_loss
					torch.save(model.state_dict(), 'model_states/epoch_{0}_{1}.pt'.format(epoch, iteration))

				epochs.append(epoch)
				all_train_loss.append(train_loss.item())
				all_eval_loss.append(eval_loss)
				all_lr.append(scheduler.optimizer.param_groups[0]['lr'])

	plot({
		'epoch': epochs,
		'train loss': all_train_loss,
		'valid loss': all_eval_loss,
		'learning rate': all_lr
	})
			


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
		eval_loss /= len(dataloader)

		#print(model.inference('Tell me about yourself and what you like to do most in this world?'))

	return eval_loss


def plot(data: dict):
	num_plots = len(data)

	for index, (title, values) in enumerate(data.items()):
		plt.subplot(1, num_plots, index + 1)
		plt.plot(range(len(values)), values)
		plt.title(title)

		if title == 'learning rate':
			plt.yscale("log")

	plt.tight_layout()
	plt.savefig('training_plots.png')
	plt.show()
