from torch import Tensor, nn
import torch.nn.functional as F
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead=8, dmodel=512):
        super().__init__()
        dk = dmodel//nhead
        self.nhead = nhead
        self.dmodel = dmodel
    
        self.WQ = torch.nn.Parameter(torch.rand((nhead,dmodel,dk),requires_grad=True, device=torch.device('cuda:0')))
        self.WK = torch.nn.Parameter(torch.rand((nhead,dmodel,dk),requires_grad=True, device=torch.device('cuda:0')))
        self.WV = torch.nn.Parameter(torch.rand((nhead,dmodel,dk),requires_grad=True, device=torch.device('cuda:0')))
        self.WO = nn.Linear(dk*nhead, dmodel)

        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.WQ.data.uniform_(-initrange, initrange)
        self.WK.data.uniform_(-initrange, initrange)
        self.WV.data.uniform_(-initrange, initrange)

    def forward(self, K:Tensor, Q:Tensor, V:Tensor, mask:Tensor):
        #  Q: [N, Q_len, dmodel]
        #  K: [N, K_len, dmodel]
        #  V: [N, V_len, dmodel]
        #  K_len == V_len
        N, q_len, dmodel = Q.shape
        k_len = K.shape[1]
        Q = torch.unsqueeze(Q,1)            #  Q: [N, 1, Q_len, dmodel]
        K = torch.unsqueeze(K,1)            #  K: [N, 1, K_len, dmodel]
        V = torch.unsqueeze(V,1)            #  V: [N, 1, V_len, dmodel]
        
        Qi:Tensor= Q @ self.WQ              # Qi: [N, nheads, Q_len, dk]
        Ki:Tensor = K @ self.WK             # Ki: [N, nheads, K_len, dk]
        Vi:Tensor = V @ self.WV             # Vi: [N, nheads, V_len, dk]
        dk = Qi.shape[-1]

        energy = Qi@Ki.contiguous().transpose(-2,-1) # [N, nheads, Q_len, K_len]

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e20)

        headi = F.softmax( energy / math.sqrt(float(dk)), dim=-1) # headi: [N, nheads, Q_len, K_len]
        headi:Tensor = headi@Vi                # [N, nheads, Q_len, dk]  (ps. k_len==v_len, 在该维度计算)
        output = self.WO(headi.transpose(1,2).reshape(N, q_len, dk*self.nhead))
        return output           #[N, q_len, dmodel]

class PosEncoder(nn.Module):

    def __init__(self, dmodel: int, dropout: float = 0.1, max_len: int = 515):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel))
        pe = torch.zeros(max_len, dmodel)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [N, seq_len, embedding_dim]
        pe : [max_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, nhead=8, dmodel=512, forward_expansion=4, p=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(nhead, dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.FFnet = nn.Sequential(
            nn.Linear(dmodel, forward_expansion*dmodel),
            nn.Linear(forward_expansion*dmodel, dmodel)
        )
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        at = self.dropout(self.attention(x,x,x,mask))
        x = self.norm1(at+x)

        at = self.dropout(self.FFnet(x))
        x = self.norm2(at+x)
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, nlayer=6, nhead=8, dmodel=512, forward_expansion=4, p=0.1, max_len:int=515):
        super().__init__()
        self.word_emb = nn.Embedding(src_vocab_size, dmodel)
        self.add_pos= PosEncoder(dmodel=dmodel,max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(nhead, dmodel, forward_expansion, p) for i in range(nlayer)])
    
    def forward(self, x, mask=None):
        # x: [N, seq_len]
        x = self.add_pos(self.word_emb(x))  # x:[N, seq_len, dmodel]
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, nhead=8, dmodel=512, forward_expansion=4, p=0.1):
        super().__init__()
        self.attention_masked = MultiHeadAttention(nhead, dmodel)
        self.norm1 = nn.LayerNorm(dmodel)

        self.attention= MultiHeadAttention(nhead, dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

        self.FFnet = nn.Sequential(
            nn.Linear(dmodel, forward_expansion*dmodel),
            nn.ReLU(),
            nn.Linear(forward_expansion*dmodel, dmodel),
            nn.ReLU(),
        )
        self.norm3 = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(p)

    def forward(self, k, q, v, src_mask, tar_mask):
        at = self.dropout(self.attention_masked(q, q, q, tar_mask))
        x = self.norm1(at+q)

        at = self.dropout(self.attention(k, x, v, src_mask))
        x = self.norm2(at+x)

        at = self.dropout(self.FFnet(x))
        x = self.norm3(at+x)
        return x

class Decoder(nn.Module):
    def __init__(self, tar_vocab_size, nlayer=6, nhead=8, dmodel=512, forward_expansion=4, p=0.1, max_len:int=515):
        super().__init__()
        self.word_emb = nn.Embedding(tar_vocab_size, dmodel)
        self.add_pos= PosEncoder(dmodel=dmodel,max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(nhead, dmodel, forward_expansion, p) for i in range(nlayer)])
        self.output = nn.Linear(dmodel, tar_vocab_size) #? 这个参数要跟共享吗？？？？
    
    def forward(self, q, enc, src_mask, tar_mask):
        # q: [N, seq_len]
        q = self.add_pos(self.word_emb(q))  # q:[N, seq_len, dmodel]
        for layer in self.layers:
            q = layer(enc, q, enc, src_mask, tar_mask)
        q = self.output(q)   # q:[N, seq_len, tar_vocab_size]
        return F.log_softmax(q, dim=-1)

class Transformer(nn.Module):
    def __init__(self, device, max_len, src_vocab_size, tar_vocab_size, pad_id=0, nlayer=6, nhead=8, dmodel=512, forward_expansion=4, p=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, nlayer, nhead, dmodel, forward_expansion, p, max_len)
        self.decoder = Decoder(tar_vocab_size, nlayer, nhead, dmodel, forward_expansion, p, max_len)
        self.device = device
        self.pad_id = pad_id

    def make_tar_mask(self, tar:Tensor):
        N, tar_len = tar.shape
        tar_mask = torch.tril(torch.ones(tar_len, tar_len)).unsqueeze(0).expand(N, -1, -1).unsqueeze(1).to(self.device)
        # tar_mask [N, 1, tar_len, tar_len]
        return tar_mask

    def make_src_mask(self, src:Tensor):
        N, src_len = src.shape
        src_mask = (src!=self.pad_id).unsqueeze(1).unsqueeze(1)       # [N, 1, 1, src_len]
        return src_mask

    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(tar)
        enc = self.encoder(src, src_mask)
        out = self.decoder(tar, enc, src_mask, tar_mask)   # [N, seq_len, tar_vocab_size]
        return out
