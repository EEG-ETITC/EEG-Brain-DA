from torch import nn
from torch import stack
import torch



class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes, dropout_layer:float=0.1, conv_2d_layer:int=32, linear_layer_classifier: int= 64, 
                 cnn_out=32, lstm_hidden=128, lstm_layers=2):
        super().__init__()

        # --- CNN para procesar espacialmente los 19x512 ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv_2d_layer, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(conv_2d_layer),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Conv2d(conv_2d_layer, cnn_out, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(cnn_out),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))   # Reduce (19,512) -> (1,1)
        )

        # --- LSTM para procesar la secuencia temporal ---
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )

        # --- Clasificador final ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, linear_layer_classifier),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Dropout(dropout_layer),
            nn.Linear(linear_layer_classifier, num_classes)
        )

    def forward(self, x):
        """
        x: tensor de forma [Batch, T: patches, Channels, Frequency]
        """
        
        B, T, C, F = x.shape
        cnn_features = []

        # Paso 1: procesar cada instante temporal con la CNN
        for t in range(T):
            # x[:, t]: [B, C, F]
            xt = x[:, t].unsqueeze(1)  # -> [B, 1, C, F]
            feat_t = self.cnn(xt)      # -> [B, cnn_out, 1, 1]
            # print(feat_t.view(B, -1).shape)
            feat_t = feat_t.view(B, -1)  # -> [B, cnn_out]
            cnn_features.append(feat_t)

        # Paso 2: reconstruir la secuencia temporal
        seq = stack(cnn_features, dim=1)  # [B, T, cnn_out]

        # Paso 3: procesar con LSTM
        lstm_out, (h, c) = self.lstm(seq)       # [B, T, lstm_hidden]
        # Paso 4: usar la última salida de la secuencia
        last_output = lstm_out[:, -1, :]        # [B, lstm_hidden]

        # Paso 5: clasificar
        out = self.classifier(last_output)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        """
        lstm_out: [B, T, H]
        """
        scores = self.attn(lstm_out)       # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context, weights