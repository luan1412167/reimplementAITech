
from os import device_encoding
from random import shuffle
from turtle import forward
import torch

from dataset import IMDBBertDataset

class JointEmbedding(nn.Module):

    def __init__(self, vocal_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size
        self.tokem_emb = nn.Embedding(vocal_size, size)
        self.segment_emb = nn.Embedding(vocal_size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, input_tensor)
        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size // 2 + 1:] = 1
        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)
        pos = torch.arage(sentence_size, dtype=torch.long).to(device)
        d= torch.arange(dim, dtype=torch.long).to(device)
        d = (2*d/dim)
        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)
        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)

class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()
        self.dim_inp = dim_inp
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)
        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out) -> None:
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)

class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1) -> None:
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            dd.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)


    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):
    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads=3) -> None:
        super(BERT, self).__init__()
        self.embedding = JointEmbedding(vocab_size, dim_inp)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)
        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(dim_inp, 2)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        encoded = self.encoder(embedded, attention_mask)
        token_predictions = self.token_prediction_layer(encoded)
        first_word = encoded[:, 0 ,:]
        return self.softmax(token_predictions), self.classification_layer(first_word)


class BertTrainer:
    def __init__(self, model: BERT,
                    dataset: IMDBBertDataset,
                    log_dir: Path,
                    checkpoint_dir: Path = None,
                    print_progress_every: int = 10,
                    print_accuracy_every: int = 50,
                    batch_size: int = 24,
                    learning_rate: float = 0.005,
                    epochs: int = 5
                    ) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.ml_criterion = nn.NLLLoss(ignore_index=0).tod(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)
        

    def train(self, epoch: int):
        print("Training epoch: ", epoch)
        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, value in enumerate(self.loader):
            index = i + 1
            inp, mask, inverse_token_mask, token_target, nsp_target = value
            self.optimizer.zero_grad()
            token, nsp = self.model(inp, mask)
            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill(tm, 0)
            loss_token = self.ml_criterion(token.transpose(1, 2), token_target)
            loss_nsp = self.criterion(nsp, nsp_target)
            loss = loss_nsp + loss_token
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_token

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev) 
                s = self.training_summary(elapsed, index, average_nsp_loss, average_mlm_loss)
                if index % self._accuracy_every == 0:
                    s += self.accuracy_summary(index, token, nsp, token_target, nsp_target)
                print(s)

                average_nsp_loss = 0
        average_mlm_loss = 0
        return loss