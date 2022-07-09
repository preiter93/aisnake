import torch
import torch.nn as nn

# import torch.nn.functional as F
from copy import deepcopy
import random
import os


class NeuralNet4Layer(nn.Module):
    """
    N-Layer neural network
    """

    def __init__(self, layer_sizes: [int], bias=False):
        """
        Initialize with random weights.

        :layer_size: Layer sizes, order [input, hidden1, hidden2, output]
        :bias: If False, neglects bias.
        """
        assert len(layer_sizes) == 4, len(layer_sizes)
        super().__init__()
        self.layer_sizes = layer_sizes
        self.depth = len(layer_sizes)
        self.linear1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.linear2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.linear3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.linear = [self.linear1, self.linear2, self.linear3]
        self.bias = bias
        # Number of weights per layer
        self.num_weights = [
            self.layer_sizes[i] * self.layer_sizes[i + 1] for i in range(self.depth - 1)
        ]
        self.apply(self._init_weights)

    def forward(self, x):
        """
        Evaluate Neural net

        :x: Input state in torch Tensor format
        :return: Neural net output in Tensor format

        Example
        >>>
        net = NeuralNetN([11, 4, 3])
        state = [0] * 11
        state = torch.tensor(state, dtype=torch.float)
        x = net.forward(state)
        >>>
        """
        for l in self.linear:
            # x = F.relu(l(x))
            x = l(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)
        if not os.path.isfile(file_name):
            raise ValueError("File {} not found".format(file_name))
        self.load_state_dict(torch.load(file_name))

    def _init_weights(self, module):
        """
        Initialize weights with random numbers
        between -1 and 1
        """
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-1.0, 1.0)
            if module.bias is not None and not self.bias:
                module.bias.data.zero_()
            else:
                module.bias.data.uniform_(-1.0, 1.0)


class EvolvingNeuralNet(NeuralNet4Layer):
    """
    Neural net that implements necessary
    evoluationary methods.
    """

    def __init__(self, layer_sizes: [int], bias=False):
        super().__init__(layer_sizes, bias)

    def recombine(self, other):
        """
        Recombine two neural nets and produce
        two new offsprings.

        There are many ways to recombine two nets.
        Currently each layer and each node on the
        right is seen as seperate chromosome.

        :other: Mating partner of type Neuralnet

        :return: Two offsprings (NeuralNet, NeuralNet)
        """
        return recombine_weights(self, other, bias=self.bias)

    def mutate(self, N, N_biases=2):
        from .utils import convert_index_1d_to_2d

        # Total number of weights
        total_num_weights = 0
        for i in range(len(self.layer_sizes) - 1):
            total_num_weights += self.layer_sizes[i] * self.layer_sizes[i + 1]

        for _ in range(N):
            i_gene = random.randint(0, total_num_weights - 1)

            # # Check if gene is within scope
            # if i_gene >= total_num_weights:
            #     raise IndexError(
            #         "index {} of gene out of Nnet size {}".format(i_gene, total_num_weights)
            #     )

            # Figure out at which plane of the nnet the gene lays
            depth = 0
            while (
                i_gene // (self.layer_sizes[depth] * self.layer_sizes[depth + 1])
            ) > 0:
                i_gene -= self.layer_sizes[depth] * self.layer_sizes[depth + 1]
                depth += 1
            max_row, max_col = self.linear[depth].weight.shape
            (row, col) = convert_index_1d_to_2d(i_gene, max_row, max_col)
            self.linear[depth].weight.data[row, col] = random.uniform(-1, 1)

        # Mutate biases
        if self.bias:
            total_num_biases = 0
            for i in range(len(self.layer_sizes) - 1):
                total_num_biases += self.layer_sizes[i + 1]
            for _ in range(N_biases):
                i_gene = random.randint(0, total_num_biases - 1)
                # Figure out at which plane of the nnet the gene lays
                depth = 0
                while (i_gene // self.layer_sizes[depth + 1]) > 0:
                    i_gene -= self.layer_sizes[depth + 1]
                    depth += 1
                self.linear[depth].bias.data[i_gene] = random.uniform(-1, 1)

    # def mutate(self, N):
    #     """
    #     Mutate N randomly picked weights.

    #     TODO: N unique mutations. Currently one gene might be mutated
    #     more than once.

    #     :N: Number of weights to mutate.
    #     """
    #     total_weights = 0
    #     for i in range(self.depth - 1):
    #         total_weights += self.num_weights[i]
    #     genes = [random.randint(0, total_weights - 1) for _ in range(N)]
    #     for gene in genes:
    #         n, i, j = self._pos_in_layer(gene)
    #         self.linear[n].weight.data[i, j] = random.uniform(-1, 1)

    # def _pos_in_layer(self, n: int):
    #     """
    #     Given the absolute position n (< total size of Nnet), find
    #     the position where it belongs
    #     """
    #     i, p, w = 0, 0, self.num_weights[0]
    #     while i < self.depth - 1:
    #         if n < w:
    #             return i, (n - p) // self.layer_sizes[i], (n - p) % self.layer_sizes[i]
    #         else:
    #             i += 1
    #             p += self.num_weights[i - 1]
    #             w += self.num_weights[i]
    #     raise ValueError("N {:6} outside Neural net scope.".format(n))


def recombine_outgoing(net_a, net_b):
    """
    Recombine outgoing weights.

    Pick a node on the LEFT of the layer, recombine all
    outgoing connections of net_a and net_b. Then
    pick the next node and to the same again. For each
    node choose a new random number that specifies the
    crossover point lying in between [1, len(outgoing) -2].

    :net_a: Neural network A
    :net_b: Neural network B
    :return: 2 offspings of type [Neuralnetwork; 2]
    """
    # # Check input
    # if not isinstance(net_a, NeuralNet) or not isinstance(net_b, NeuralNet):
    #     raise ValueError("Wrong input.")

    # Initiate Offsprings
    off_a = deepcopy(net_a)
    off_b = deepcopy(net_a)

    # Recombine
    with torch.no_grad():
        for n in range(net_a.depth - 1):
            num_genes = net_a.layer_sizes[n + 1]

            # Consider multiple individual chromosomes
            for i in range(net_a.layer_sizes[n]):
                pt = random.randint(1, num_genes - 2)
                off_a.linear[n].weight[pt:, i] = net_b.linear[n].weight[pt:, i]
                off_b.linear[n].weight[:pt, i] = net_b.linear[n].weight[:pt, i]

    return [off_a, off_b]


def recombine_ingoing(net_a, net_b):
    """
    Recombine ingoing weights.

    Pick a node on the RIGHT of the layer, recombine all
    outgoing connections of net_a and net_b. Then
    pick the next node and to the same again. For each
    node choose a new random number that specifies the
    crossover point lying in between [1, len(incoming) -2].

    :net_a: Neural network A
    :net_b: Neural network B
    :return: 2 offspings of type [Neuralnetwork; 2]
    """
    # # Check input
    # if not isinstance(net_a, NeuralNet) or not isinstance(net_b, NeuralNet):
    #     raise ValueError("Wrong input.")

    # Initiate Offsprings
    off_a = deepcopy(net_a)
    off_b = deepcopy(net_a)

    # Recombine
    with torch.no_grad():
        for n in range(net_a.depth - 1):
            num_genes = net_a.layer_sizes[n]

            # Consider multiple individual chromosomes
            for i in range(net_a.layer_sizes[n + 1]):
                pt = random.randint(1, num_genes - 2)
                off_a.linear[n].weight[i, pt:] = net_b.linear[n].weight[i, pt:]
                off_b.linear[n].weight[i, :pt] = net_b.linear[n].weight[i, :pt]

    return [off_a, off_b]


def recombine_weights(net_a, net_b, bias=False):
    """
    Recombine weights randomly.

    Pick an arbitrary number that is 0 or 1. If 0, first
    offspring inherhits from first parent, if 1, it inherits
    from second parent.

    :net_a: Neural network A
    :net_b: Neural network B
    :bias: If true, also reombine bias values
    :return: 2 offspings of type [Neuralnetwork; 2]
    """
    # # Check input
    # if not isinstance(net_a, NeuralNet) or not isinstance(net_b, NeuralNet):
    #     raise ValueError("Wrong input.")

    # Initiate Offsprings
    off_a = deepcopy(net_a)
    off_b = deepcopy(net_a)

    # Recombine
    with torch.no_grad():
        for n in range(net_a.depth - 1):
            for i in range(net_a.layer_sizes[n]):
                for j in range(net_a.layer_sizes[n + 1]):
                    pt = random.randint(0, 1)
                    if pt == 0:
                        off_b.linear[n].weight[j, i] = net_b.linear[n].weight[j, i]
                    else:
                        off_a.linear[n].weight[j, i] = net_b.linear[n].weight[j, i]

                if bias:
                    for i in range(net_a.layer_sizes[n + 1]):
                        pt = random.randint(0, 1)
                        if pt == 0:
                            off_b.linear[n].bias[i] = net_b.linear[n].bias[i]
                        else:
                            off_a.linear[n].bias[i] = net_b.linear[n].bias[i]
    return [off_a, off_b]
