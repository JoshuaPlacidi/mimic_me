import sys

from dataset import create_dataloaders
from model import ChatModel
from training import train

# read chat foldered arugment and initialise chat dict
# if len(sys.argv) != 2:
# 	raise Exception('Incorrect use, must include argument for chat file path location: python3 create_dataset.py file_path')
# folder_path = sys.argv[1]

folder_path = '/Users/joshua/env/datasets/whatsapp_chat_logs'

train_dataloader, test_dataloader = create_dataloaders('joshua', folder_path, batch_size=1)

model = ChatModel()

train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader)

prompt = 'Do you like dogs?'
answer = model.inference(prompt)

print(answer)