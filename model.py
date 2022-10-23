from os import truncate
import torch
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizerFast

class ChatModel(torch.nn.Module):
	def __init__(self):
		'''
		Description:
			Model object, composed of pretrained transformer encoder and transformer decoder modules
		'''
		super(ChatModel, self).__init__()

		# initialise tokenizer and download pretrained weights
		self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

		# initialise model and download pretrained weights
		self.model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
		
		
		self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
		self.model.config.eos_token_id = self.tokenizer.sep_token_id
		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		self.model.config.vocab_size = self.model.config.encoder.vocab_size

		self.model.config.max_length = 142
		self.model.config.min_length = 56
		self.model.config.no_repeat_ngram_size = 3
		self.model.config.early_stopping = True
		self.model.config.length_penalty = 2.0
		self.model.config.num_beams = 4


	def encode(self, text: list):
		'''
		Description:
			Batch convert a str() into a sequence of tokens representing the original text

		Params:
			- text: list of strings containing sequences to be encoded

		Returns:
			- batch_encoding: PyTorch tensor of token sequence for each text element input
		'''
		batch_encoding = self.tokenizer(
			text=list(text),
			padding='max_length',
			max_length=100,
			truncation=True,
			return_tensors='pt',
		)

		return batch_encoding

	def forward(self, prompt_tkn, answer_tkn):
		'''
		Description:
			Forward pass through the model

		Params:
			- prompt_tkn: batch of token sequences
			- answer_tkn: batch of corresponding answer sequences
		'''
		x = self.model(input_ids=prompt_tkn, labels=answer_tkn)
		return x

	def inference(self, promt_str):
		with torch.no_grad():

			prompt_tkn = self.encode(promt_str)['input_ids']

			answer_tkn = self.model.generate(prompt_tkn, bos_token_id = prompt_tkn.shape[-1])

			answer_str = self.tokenizer.decode(answer_tkn[0])

			return answer_str