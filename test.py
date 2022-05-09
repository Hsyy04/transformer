from aiohttp import TraceDnsCacheHitParams
from transformer import Transformer
import torch
from main import SentPairSet
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
LEARNING_RATE = 0.00003
EPOCH_NUM = 40
BATCH_SIZE= 1
HEAD_NUM = 4
LAYER_NUM = 2
DMODEL = 256
TAR_LEN = 128
SRC_LEN = 128
data_set = SentPairSet(SRC_LEN, TAR_LEN)

src = "I love you!"
tar = ""
model = torch.load("model/model(2_2_256)_lr3e-05_en40_bsz1.pth")
model.to(device)
model.eval()

id_src, id_tar = data_set.get_id(src, tar)
src_input = torch.Tensor(id_src).to(int).unsqueeze(0).to(device)
tar_input = torch.Tensor(id_tar).to(int).unsqueeze(0).to(device)

output = 0
while output != data_set.tar_word2id["<eos>"] and len(id_tar)<=10:
    out = model(src_input, tar_input)   # [1, q_len, vocab_size]
    out = out.squeeze(0)[-1,:]          # [vocab_size]
    output = torch.argmax(out)
    # new tar
    id_tar.append(output)
    tar_input = torch.Tensor(id_tar).to(int).unsqueeze(0).to(device)

print(id_tar)
str = data_set.tar_id2word[id_tar[0]]
print("\n")

for id in id_tar[1:]:
    str+=data_set.tar_id2word[id.item()]
print(str)