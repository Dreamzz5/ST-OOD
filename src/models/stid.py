from src.base.model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(BaseModel):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, if_spatial=True, dropout=0, **args):
        super(STID, self).__init__(**args)
        # attributes
        self.num_nodes = self.node_num
        self.node_dim = 24
        self.input_len = self.horizon
        self.input_dim = self.input_dim
        self.embed_dim = 32
        self.output_len = 12
        self.num_layer = 1
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        self.time_of_day_size = 288
        self.day_of_week_size = 7
        self.dropout = dropout

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = if_spatial

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # encoding
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, history_data, labels) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        # Normalization from Non-stationary Transformer
        # means = history_data[..., 0:1].mean(1, keepdim=True).detach()
        # history_data[..., 0:1] = history_data[..., 0:1] - means
        # stdev = torch.sqrt(torch.var(history_data[..., 0:1], dim=1, keepdim=True, unbiased=False) + 1e-5)
        # history_data[..., 0:1] /= stdev
        
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :]).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        time_series_emb = self.time_series_emb_layer(input_data)

        node_embs = []
        if self.if_spatial:
            # expand node embeddings
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            node_embs.append(F.dropout(node_emb, self.dropout, training=self.training))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_embs + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        # prediction = prediction * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1, 1))
        # prediction = prediction + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1, 1))
        

        return prediction
