from os import truncate
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class ChatModel(torch.nn.Module):
	def __init__(self):
		'''
		Description:
			Model object, composed of pretrained transformer encoder and transformer decoder modules
		'''
		super(ChatModel, self).__init__()

		self.model_name = "facebook/blenderbot-400M-distill"

		self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name, truncation_side='left')

		self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)

	def encode(self, text: str):
		'''
		Description:
			Convert a str() into a sequence of tokens representing the original text

		Params:
			- text: list of strings containing sequences to be encoded

		Returns:
			- tokens (torch.Tensor): token tensor for each text element input
			- mask (torch.Tensor): tensor of 1s and 0s, 1 where attention should be calculated, 0 for tokens that should be ignored
		'''

		output = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs = text,
			return_tensors='pt',
			padding='max_length',
			max_length=100,
			truncation=True)

		tokens, mask = output['input_ids'], output['attention_mask']

		return tokens, mask


	def forward(self, prompt_tokens, prompt_mask, labels):
		'''
		Description:
			Forward pass through the model

		Params:
			- prompt_tkn: batch of token sequences
			- answer_tkn: batch of corresponding answer sequences

		Returns:
			- x (transformers.modeling_outputs.Seq2SeqModelOutput): output of the forward pass
		'''
		x = self.model(input_ids=prompt_tokens, attention_mask=prompt_mask, labels=labels)
		return x

	def inference(self, context: str):
		'''
		Description:
			Generate a string output for a given context string input

		Params:
			- context (str): the input text to condition on

		Returns:
			- response (str): natural language response to the context
		
		'''
		with torch.no_grad():

			context_tokens, context_mask = self.encode([context])

			response_tokens = self.model.generate(input_ids=context_tokens, attention_mask=context_mask)

			response = self.tokenizer.batch_decode(response_tokens, skip_special_tokens=True)[0]

			return response
