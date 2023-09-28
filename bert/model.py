

class JointEmbedding(nn.Module):

    def __init__(self, vocal_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size
        self.tokem_emb = nn.Embedding(vocal_size, size)
        self.segment_emb = nn.Embedding(vocal_size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)