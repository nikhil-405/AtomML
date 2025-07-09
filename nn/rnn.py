from core.tensor import Tensor
from nn.module import BaseModule
import numpy as np

# https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/rnn.py#L469
# I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, Chapter 10: Sequence Modeling: Recurrent and Recursive Nets

class RNN(BaseModule):
    def __init__(self, input_size, hidden_size, num_layers = 1, bidirectional = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.W_ih = [] # input to hidden
        self.W_hh = []
        self.b_h  = []

        for layer in range(num_layers):
            layer_W_ih = []
            layer_W_hh = []
            layer_b_h = []

            for direction in range(self.num_directions):
                in_dim = input_size if layer == 0 else hidden_size * self.num_directions

                w_ih = Tensor(np.random.randn(in_dim, hidden_size), requires_grad = True)
                w_hh = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad = True)
                b_h = Tensor(np.zeros(hidden_size), requires_grad = True)

                # add to params
                self.add_param(w_ih)
                self.add_param(w_hh)
                self.add_param(b_h)

                layer_W_ih.append(w_ih)
                layer_W_hh.append(w_hh)
                layer_b_h.append(b_h)

            self.W_ih.append(layer_W_ih)
            self.W_hh.append(layer_W_hh)
            self.b_h.append(layer_b_h)

    def forward(self, x_np):
        x = Tensor(x_np, requires_grad=False)
        batch, seq_len, _ = x.data.shape # seq_len = T

        h = [[Tensor(np.zeros((batch, self.hidden_size)), requires_grad = False)
             for _ in range(self.num_directions)]
             for _ in range(self.num_layers)]

        # iterate over time steps
        for t in range(seq_len):
            x_t = x[:, t]

            for layer in range(self.num_layers):
                new_h = [None] * self.num_directions

                for direction in range(self.num_directions):
                    w_ih = self.W_ih[layer][direction]
                    w_hh = self.W_hh[layer][direction]
                    b_h = self.b_h[layer][direction]

                   
                    if layer == 0:
                        inp = x_t
                   
                    else:
                        prev = h[layer-1]
                        inp = Tensor(np.concatenate([p.data for p in prev], axis=1), requires_grad = True)

                    if direction == 1 and self.num_directions == 2:
                        seq_index = seq_len - 1 - t
                        inp = x[:, seq_index]

                    h_prev = h[layer][direction]
                    h_new = (inp @ w_ih + h_prev @ w_hh + b_h).tanh()
                    new_h[direction] = h_new

                h[layer] = new_h


        last_layer_h = h[-1]

        if self.bidirectional:
            out = Tensor(np.concatenate([hh.data for hh in last_layer_h], axis = 1), requires_grad = True)
        else:
            out = last_layer_h[0]

        return out
