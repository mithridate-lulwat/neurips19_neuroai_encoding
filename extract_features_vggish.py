
from torchvggish.torchvggish import vggish_input, vggish
import torch
import time
assert torch.cuda.memory_allocated() == 0
example = vggish_input.wavfile_to_examples("sherlockaudio.wav")
print("after loading  {} available memory".format(torch.cuda.memory_allocated()))
print('examples : ',example.shape)
DEVICE = torch.device("cuda:0")

t = time.time()
embedding_model = vggish().to(DEVICE)
f = time.time()
print("{:2f}".format(f-t))
embeddings = embedding_model.forward(example)
print("embeddings : ",embeddings.shape)

