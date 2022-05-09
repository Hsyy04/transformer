from tokenize import String
from torch import int32, int64, kl_div
from sklearn.utils import shuffle
from torch.optim.optimizer import Optimizer
from transformer import Transformer
from torch.utils.data import DataLoader,Dataset,random_split
from torch import Tensor,nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from datetime import datetime
from tqdm import tqdm
import jieba
import os
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class SentPairSet(Dataset):
    def __init__(self, src_len, tar_len) -> None:
        super().__init__()
        self.src_len = src_len
        self.tar_len = tar_len

        with open('data/dict_en.txt','r', encoding='UTF-8') as f:
            words_src_all=f.read().split('\n')
        self.src_word2id = dict([(w,i) for i,w in enumerate(words_src_all)])
        self.src_id2word = dict([(i,w) for i,w in enumerate(words_src_all)])
        
        with open('data/dict_ch.txt','r', encoding='UTF-8') as f:
            words_tar_all = f.read().split('\n')
        self.tar_word2id = dict([(w,i) for i,w in enumerate(words_tar_all)])
        self.tar_id2word = dict([(i,w) for i,w in enumerate(words_tar_all)])

        with open("data/sent_pairs.txt", 'r', encoding='UTF-8') as f:
            data_raw = f.read().split('\n')
        self.data_sent_pair = []
        for sent in data_raw[:1000]:
            sent_src_id, sent_tar_id = self.get_id(sent.split('\t')[0], sent.split('\t')[1])
            self.data_sent_pair.append([sent_src_id[:src_len], sent_tar_id[:tar_len]])
        
    def __getitem__(self, index):
        return Tensor(self.data_sent_pair[index][0]).to(int32),Tensor(self.data_sent_pair[index][1]).to(int64)

    def __len__(self):
        return len(self.data_sent_pair)

    def get_src_vocab_size(self):
        return len(self.src_word2id)
    def get_tar_vocab_size(self):
        return len(self.tar_word2id)
    def get_id(self, st_src, st_tar):
        # source sentence
        sent_src = st_src.split(' ')
        sent_src_id = []
        for wd in sent_src:
            if wd in self.src_word2id:
                sent_src_id.append(self.src_word2id[wd])
            else: 
                sent_src_id.append(self.src_word2id['<unk>'])
        # target sentence
        sent_tar_id = []
        sent_tar_id.append(self.tar_word2id['<start>'])
        if len(st_tar)!=0:
            sent_tar = jieba.lcut(st_tar)
            for wd in sent_tar:
                if wd in self.tar_word2id:
                    sent_tar_id.append(self.tar_word2id[wd])
                else: 
                    sent_tar_id.append(self.tar_word2id['<unk>'])
            sent_tar_id.append(self.tar_word2id['<eos>'])
            # add padding
            while(len(sent_src_id)<self.src_len): sent_src_id.append(0)
            while(len(sent_tar_id)<self.tar_len): sent_tar_id.append(0)

        return sent_src_id, sent_tar_id

class LabelSmoothing(nn.Module):
    # FIXME
    def __init__(self, tar_len, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.tar_len = tar_len
        self.true_dist = None

    def forward(self, x, target):
        # x: [N, seq_len, vocab_size]
        # target: [N, seq_len]  this is true
        target  = target.fill(self.smoothing / (self.tar_len - 2))
        target = target.scatter(1, target.data.unsqueeze(1), self.confidence)
        target[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            target = target.index_fill(0, mask.squeeze(), 0.0)

        return self.criterion(x, Variable(target, requires_grad=False))

def forward_and_loss(model, data_batch, mode, optimizer:Optimizer=None, smooth_loss=None):
    if mode == "train":
        model.train()
    else: model.eval()
    tot_loss = 0.0
    for src, tar in tqdm(data_batch, mininterval=10):
        src, tar = src.to(device), tar.to(device)
        output = model(src, tar)
        if smooth_loss is not None:
            loss = smooth_loss(output, tar)
            tot_loss += loss
        else: 
            N, len, vocab= output.shape
            tar_p = torch.zeros((N, len, vocab), dtype=torch.float, device=device).scatter_(dim=-1, index=tar.unsqueeze(-1), value=1)
            loss = F.kl_div(output, tar_p)
            tot_loss += loss
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return tot_loss
if __name__ == "__main__":
    LEARNING_RATE = 0.00003
    EPOCH_NUM = 60
    BATCH_SIZE= 1
    HEAD_NUM = 4
    LAYER_NUM = 2
    DMODEL = 256
    TAR_LEN = 128
    SRC_LEN = 128
    currentTime = datetime.now().strftime('%b%d_%H-%M-%S')
    NAME = f'model({LAYER_NUM}_{LAYER_NUM}_{DMODEL})_lr{LEARNING_RATE}_en{EPOCH_NUM}_bsz{BATCH_SIZE}'
    # data
    data_set = SentPairSet(SRC_LEN, TAR_LEN)
    train_len = int(len(data_set)*0.8)
    train_dataset, test_dataset = random_split(
        dataset=data_set,
        lengths=[train_len, len(data_set)-train_len],
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # model and optimizer
    model = Transformer(
        device, 1024, 
        data_set.get_src_vocab_size(), 
        data_set.get_tar_vocab_size(),
        nlayer=LAYER_NUM,
        nhead=HEAD_NUM,
        dmodel=DMODEL
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #? smooth loss
    # FIXME: smooth_loss = LabelSmoothing(TAR_LEN, 0, 0.5)
    # tensorboard
    writer = SummaryWriter(f"runs/{NAME}_{currentTime}/")
    # train
    print("start trainning...")
    for epoch in range(EPOCH_NUM):
        model.train()
        print(f"Epoch: {epoch+1}/{EPOCH_NUM}")

        totloss = forward_and_loss(model, train_dataloader,'train',optimizer)

        print(f"loss:{totloss}")
        writer.add_scalars('train/loss', {'loss':totloss}, epoch)
        # if epoch % 10 == 0:
        #     test_loss = forward_and_loss(model, test_dataloader, 'test')
        #     print(f"test_loss:{test_loss}")
        #     dic = {'test':test_loss}
        #     writer.add_scalars('result/loss', dic, epoch)
        #     torch.save(model, f"model/{NAME}.pth") 
        #     print("saving...")
    # TODO: bleu score
    torch.save(model, f"model/{NAME}.pth") 
    print('\n')