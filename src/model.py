import torch
import torch.nn as nn

from torch.nn.utils.rnn import(
    pack_padded_sequence,
    pad_packed_sequence
)

#configuring the model
VOCAB_SIZE = 50000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 5
DROPOUT = 0.3

class PunctuationBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_DIM*2,
            NUM_CLASSES
        )


    def forward(self, input_ids, lengths):

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1)
        )

        lstm_output = self.dropout(lstm_output)

        logits = self.classifier(lstm_output)

        return logits

model = PunctuationBiLSTM()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Model created successfully!")
print("Total trainable parameters:", total_parameters)

sample_input = torch.randint(
    low=1,
    high=VOCAB_SIZE,
    size=(32, 128)
)

sample_lengths = torch.full(
    (32,),
    128,
    dtype=torch.long
)

with torch.no_grad():

    sample_output = model(
        sample_input,
        sample_lengths
    )

print("Input shape:", sample_input.shape)
print("Output shape:", sample_output.shape)
