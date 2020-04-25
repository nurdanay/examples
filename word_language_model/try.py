###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
					help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
					help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
					help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
					help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
					help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
					help='reporting interval')
parser.add_argument('--strategy',
					help='generation method to use')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
	parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
	model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
_SOS_token = 1

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
	hidden = model.init_hidden(1)

if args.strategy == 'sampling' :	
	input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

elif args.strategy == 'greedy' :
	input = torch.ones(1, 1, dtype=torch.long).to(device) * _SOS_token
	all_tokens = torch.zeros([0], dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
	with torch.no_grad():  # no tracking history
		for i in range(args.words):

			if is_transformer_model:
				output = model(input, False)
				word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
				
				if args.strategy == 'sampling' :
					word_idx = torch.multinomial(word_weights, 1)[0]  # does sampling
					# a multi-dimensional matrix containing elements of a single data type.
					word_tensor = torch.Tensor([[word_idx]]).long().to(device) 

					# Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
					input = torch.cat([input, word_tensor], 0)

				elif args.strategy == 'greedy' :
					# Obtain most likely word token and its softmax score
					word_idx = torch.max(word_weights, dim=1)
					# Record token
					all_tokens = torch.cat((all_tokens, input), dim=0)
					# Prepare current token to be next decoder input (add a dimension)
					input = torch.unsqueeze(input, 0)



			else:
				output, hidden = model(input, hidden)
				word_weights = output.squeeze().div(args.temperature).exp().cpu()

				if args.strategy == 'sampling' :
					word_idx = torch.multinomial(word_weights, 1)[0]  # does sampling
					input.fill_(word_idx) 

					# Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
					input = torch.cat([input, word_tensor], 0)

				elif args.strategy == 'greedy' :
					# Obtain most likely word token and its softmax score
					word_idx = torch.max(word_weights, dim=1)
					# Record token
					input = input.fill_(word_idx)
					# Prepare current token to be next decoder input (add a dimension)
					input = torch.unsqueeze(input, 0)
				

			word = corpus.dictionary.idx2word[word_idx]

			outf.write(word + ('\n' if i % 20 == 19 else ' '))

			if i % args.log_interval == 0:
				print('| Generated {}/{} words'.format(i, args.words))
