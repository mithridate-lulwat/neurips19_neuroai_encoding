
from torchvggish.torchvggish import vggish_input, vggish
import torch
import time
import tqdm
example = vggish_input.wavfile_to_examples("sherlockaudio.wav")
length = len(example)
print(example.shape)
complete_embedding = torch.tensor([])
embedding_model = vggish()
# Divide the input in parts of xxx samples
samples_per_part = 100
n_parts = length // samples_per_part + 1 
for part in tqdm.tqdm(range(n_parts)):
    partial_example = example[samples_per_part*part:samples_per_part*(part+1)]
    print(partial_example.shape)
    complete_embedding = torch.cat((complete_embedding, embedding_model.forward(partial_example)))

print("embeddings : ",complete_embedding.shape)
torch.save(complete_embedding, "sherlock_embedding.pt")

