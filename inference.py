from model import ChatModel
import torch
from transformers import BlenderbotTokenizer


model = ChatModel(device='cuda:0')

model.load_state_dict(torch.load('model_states/epoch_0_125.pt'))

model_name = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
 
text = ""

for _ in range(10):
	print()
	in_text = input(' - Human: ')

	if text:
		text = text + ' <s> ' + in_text
	else:
		text = in_text

	out_text = model.inference(text)


	text = text + '</s> <s>' + out_text + '</s>'

	print(' - Bot: {0}'.format(out_text))