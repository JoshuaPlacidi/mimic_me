import sys
import torch
from dataset import create_dataloaders
from model import ChatModel
from training import train

# read chat foldered arugment and initialise chat dict
# if len(sys.argv) != 2:
# 	raise Exception('Incorrect use, must include argument for chat file path location: python3 create_dataset.py file_path')
# folder_path = sys.argv[1]

folder_path = '/Users/joshua/env/datasets/whatsapp_chat_logs'

train_dataloader, test_dataloader = create_dataloaders('joshua', folder_path, batch_size=32)

model = ChatModel(device='cpu')

# model.load_state_dict(torch.load('model.pt'))
 
train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=1)

prompt = 'Should I buy a car?'
answer = model.inference(prompt)

print(answer)

prompt = 'Is the sky blue or green?'
answer = model.inference(prompt)
print(answer)

#prompt = 'Should I buy the new CPU from AMD?'
#answer = model.inference(prompt)

#print(answer)
