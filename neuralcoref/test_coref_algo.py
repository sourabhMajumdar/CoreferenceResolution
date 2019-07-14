import spacy
import torch
import socket
import os
import glob

from algorithm import Coref
from utils import (encode_distance, BATCH_SIZE_PATH, SIZE_FP,
                               SIZE_FP_COMPRESSED, SIZE_FS, SIZE_FS_COMPRESSED,
                               SIZE_GENRE, SIZE_PAIR_IN, SIZE_SINGLE_IN,SIZE_EMBEDDING)
from dataset import (NCDataset, NCBatchSampler,
    load_embeddings_from_file, padder_collate,
    SIZE_PAIR_IN, SIZE_SINGLE_IN, SIZE_EMBEDDING)
from model import Model

# Load datasets and embeddings
list_of_files = glob.glob('checkpoints/*modelranking') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
#save_path = os.path.join('checkpoints', current_time + '_' + socket.gethostname() + '_')
#save_name = "ranking"
#best_model_path = save_path + "best_model" + save_name
weights = 'weights/'
checkpoint_file = latest_file
embed_path = weights
tensor_embeddings, voc = load_embeddings_from_file(embed_path + "tuned_word")

# Construct model
print("🏝 Build model")
h1 = 1000
h2 = 500
h3 = 500
model = Model(len(voc), SIZE_EMBEDDING, h1,h2,h3, SIZE_PAIR_IN, SIZE_SINGLE_IN)
model.load_embeddings(tensor_embeddings)

print(model.state_dict)
cuda = torch.cuda.is_available()

if cuda:
	model.cuda()
#if weights is not None:
#	print("🏝 Loading pre-trained weights")
#	model.load_weights(weights)
if checkpoint_file is not None:
	print("⛄️ Loading model from", checkpoint_file)
	model.load_state_dict(torch.load(checkpoint_file) if cuda else torch.load(checkpoint_file, map_location=lambda storage, loc: storage))
	model.eval()

my_utterances = ['My Sister has a dog.','She loves him.']
coref0 = Coref()
print("Single Pair Inp,",SIZE_SINGLE_IN)
coref_clusters = coref0.one_shot_coref(utterances=my_utterances)
print("Coref Clusters are")
print(coref_clusters)