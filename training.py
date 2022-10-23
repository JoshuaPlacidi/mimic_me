import torch
from tqdm import tqdm


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.Dataset,
    test_dataloader: torch.utils.data.Dataset,
    num_epochs: int = 10,
    eval_steps: int = None,
    optimizer = torch.optim.AdamW,
    initial_learning_rate: float = 1e-4,
    device='cpu',
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

    # initialise the optimizer with the model parameters we would like to be adjusted during training
    optimizer = optimizer(
        params=model.parameters(),
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

    # list for storing training losses
    train_loss_list = []

    # used to store the best evaluation performance seen so far, initialised to large dummy value
    best_eval_loss = 1e10

    # move model to specified device
    model.to(device)


    for epoch in epoch_bar:

        for iteration, batch in enumerate(tqdm(train_dataloader, desc='Batch Progress Bar', position=1, leave=False)):
            # read each batch and perform a forward pass
            train_loss = train_forward(model=model, batch=batch)

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
                    'Iter {0}/{1} | Train Loss {2} | Eval Loss {3}'.format(
                        iteration,
                        len(train_dataloader),
                        round(sum(train_loss_list) / len(train_loss_list), 3),
                        round(eval_loss, 3),
                    )
                )

                # if this is the lowest eval loss seen so far then save the model parameters
                if eval_loss < best_eval_loss:
                    torch.save(model.state_dict(), 'model.pt')
            


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
    prompt_tkn = model.encode(prompt_str)['input_ids']
    answer_tkn = model.encode(answer_str)['input_ids']

    # pass through model
    loss = model(prompt_tkn=prompt_tkn, answer_tkn=answer_tkn)['loss']

    return loss

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
            eval_loss += train_forward(model, batch).item()

        # calculalate mean
        eval_loss /= len(dataloader)

    return eval_loss

