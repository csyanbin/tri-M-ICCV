import torch
import torch.nn as nn
from modules import SMAB_small
import os
import numpy as np


class DenseResidualLayer(nn.Module):
    """
    PyTorch like layer for standard linear layer with identity residual connection.
    :param num_features: (int) Number of input / output units for the layer.
    """
    def __init__(self, num_features):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x):
        """
        Forward-pass through the layer. Implements the following computation:

                f(x) = f_theta(x) + x
                f_theta(x) = W^T x + b

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, num_features) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, num_features) ).
        """
        identity = x
        out = self.linear(x)
        out += identity
        return out


class DenseResidualBlock(nn.Module):
    """
    Wrapping a number of residual layers for residual block. Will be used as building block in FiLM hyper-networks.
    :param in_size: (int) Number of features for input representation.
    :param out_size: (int) Number of features for output representation.
    """
    def __init__(self, in_size, out_size):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.linear3 = nn.Linear(out_size, out_size)
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Forward pass through residual block. Implements following computation:

                h = f3( f2( f1(x) ) ) + x
                or
                h = f3( f2( f1(x) ) )

                where fi(x) = Elu( Wi^T x + bi )

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, in_size) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, out_size) ).
        """
        identity = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out


class FilmAdaptationNetwork(nn.Module):
    """
    FiLM adaptation network (outputs FiLM adaptation parameters for all layers in a base feature extractor).
    :param layer: (FilmLayerNetwork) Layer object to be used for adaptation.
    :param num_maps_per_layer: (list::int) Number of feature maps for each layer in the network.
    :param num_blocks_per_layer: (list::int) Number of residual blocks in each layer in the network
                                 (see ResNet file for details about ResNet architectures).
    :param z_g_dim: (int) Dimensionality of network input. For this network, z is shared across all layers.
    """
    def __init__(self, layer, num_maps_per_layer, num_blocks_per_layer, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps_per_layer
        self.num_blocks = num_blocks_per_layer
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

        # Initialize a simple shared layer for all parameter adapters (gammas and betas)
        self.shared_layer = nn.Sequential(
            nn.Linear(self.z_g_dim, self.num_maps[0]),
            nn.ReLU(),
            nn.Linear(self.num_maps[0],8)
        )
        self.softmax = nn.Softmax(dim=1) # softmax gates

    def get_layers(self):
        """
        Loop over layers of base network and initialize adaptation network.
        :return: (nn.ModuleList) ModuleList containing the adaptation network for each layer in base network.
        """
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    z_g_dim=self.z_g_dim
                )
            )
        return layers

    def forward(self, x, mode='train', iteration=-1):
        """
        Forward pass through adaptation network to create list of adaptation parameters.
        :param x: (torch.tensor) (z -- task level representation for generating adaptation).
        :return: (list::adaptation_params) Returns a list of adaptation dictionaries, one for each layer in base net.
        """
        gate_logits = self.shared_layer(x)
        gate = self.softmax(gate_logits)

        if mode in ['train', 'val']:
            return [self.layers[layer](gate,x) for layer in range(self.num_target_layers)], gate_logits

        elif mode=='test':
            param_dict = []
            alpha_array = torch.zeros(960).cuda()
            alpha_dims = [0,64,192,448,960]
            trans_list = []
            for layer in range(self.num_target_layers):
                param, alpha, trSoftmax = self.layers[layer](gate, x, 'test') 
                param_dict.append( param )
                alpha_array[alpha_dims[layer]:alpha_dims[layer+1]] = alpha
                trans_list.append(trSoftmax )

            return param_dict, gate_logits


    def regularization_term(self):
        """
        Simple function to aggregate the regularization terms from each of the layers in the adaptation network.
        :return: (torch.scalar) A order-0 torch tensor with the regularization term for the adaptation net params.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmLayerNetwork(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmAdaptationNetwork.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, num_maps, num_blocks, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps
        self.num_blocks = num_blocks

        # Initialize a simple shared layer for all parameter adapters (gammas and betas)
        self.shared_layer_alphaC = nn.Sequential(
            nn.Linear(self.z_g_dim, self.num_maps),
            nn.Sigmoid()
        )
        self.shared_layer_Query = nn.Sequential(
            nn.Linear(self.z_g_dim, min(96,self.num_maps)),
            nn.ReLU()
        )

        # Initialize the processors (adaptation networks) and regularization lists for each of the output params
        self.gamma1_list, self.gamma1_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gamma2_list, self.gamma2_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.beta1_list, self.beta1_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.beta2_list, self.beta2_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gamma1De_list, self.beta1De_list = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gamma2De_list, self.beta2De_list = torch.nn.ParameterList(), torch.nn.ParameterList()

        self.gamma1_processors, self.gamma2_processors = torch.nn.ModuleList(), torch.nn.ModuleList()
        self.beta1_processors, self.beta2_processors = torch.nn.ModuleList(), torch.nn.ModuleList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            regularizer = torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001)

            self.gamma1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))
            self.beta1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))
            self.gamma2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))
            self.beta2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))
            
            self.gamma1_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.2), requires_grad=True))
            self.gamma2_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.1), requires_grad=True))
            self.beta1_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.2), requires_grad=True))
            self.beta2_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.1), requires_grad=True))
            
            ## Decoupled experts, no SAB applied
            self.gamma1De_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.01), requires_grad=True))
            self.gamma2De_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.01), requires_grad=True))
            self.beta1De_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.01), requires_grad=True))
            self.beta2De_list.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps,8), 0, 0.01), requires_grad=True))

            self.gamma1_processors.append(self._make_layer(num_maps))
            self.beta1_processors.append(self._make_layer(num_maps))
            self.gamma2_processors.append(self._make_layer(num_maps))
            self.beta2_processors.append(self._make_layer(num_maps))
    
    @staticmethod
    def _make_layer(size):
        """
        make multihead attention blocks
        """
        #return nn.Sequential(
        #    SAB(size, size, num_heads=int(size/16), ln=False),
        #    SAB(size, size, num_heads=int(size/16), ln=False)
        #)
        return SMAB_small(size, min(96,size), min(96,size), min(96,size), size, num_heads1=min(size//32,3), num_heads2=size//32, ln=False)

    def forward(self, gate, x, mode='train'):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        alpha = self.shared_layer_alphaC(x).view(-1) # C
        x = self.shared_layer_Query(x) # linear transform
        QG1 = x.view(1,1,-1)
        QG2 = x.view(1,1,-1)
        QB1 = x.view(1,1,-1)
        QB2 = x.view(1,1,-1)
        block_params = []
        softmaxTrans = torch.zeros(2,4,9,8).cuda() # 2 blocks, g1b1g2b2, (8+1)x8
        for block in range(self.num_blocks):
            gamma1 = self.gamma1_list[block].transpose(1,0).view(1,8,self.num_maps)
            gamma2 = self.gamma2_list[block].transpose(1,0).view(1,8,self.num_maps)
            beta1 = self.beta1_list[block].transpose(1,0).view(1,8,self.num_maps)
            beta2 = self.beta2_list[block].transpose(1,0).view(1,8,self.num_maps)
            gamma1De = self.gamma1De_list[block] # num_maps x 8
            gamma2De = self.gamma2De_list[block]
            beta1De = self.beta1De_list[block]
            beta2De = self.beta2De_list[block]
            alpha = alpha.view(-1)
            
            if mode=='test':
                gamma1Trans, softG11, softG12 = self.gamma1_processors[block](QG1,gamma1,'test') # test mode, return softmax
                beta1Trans,  softB11, softB12 = self.beta1_processors[block](QB1,beta1, 'test') # test mode, return softmax
                gamma2Trans, softG21, softG22 = self.gamma2_processors[block](QG2,gamma2,'test') # test mode, return softmax
                beta2Trans,  softB21, softB22 = self.beta2_processors[block](QB2,beta2, 'test') # test mode, return softmax

                ## save softmax transformer params
                softmaxTrans[block][0] = torch.cat([softG11.mean(0).view(8,8),softG12.mean(0).view(1,8)], dim=0) # 9x8
                softmaxTrans[block][1] = torch.cat([softB11.mean(0).view(8,8),softB12.mean(0).view(1,8)], dim=0) # 9x8
                softmaxTrans[block][2] = torch.cat([softG21.mean(0).view(8,8),softG22.mean(0).view(1,8)], dim=0) # 9x8
                softmaxTrans[block][3] = torch.cat([softB21.mean(0).view(8,8),softB22.mean(0).view(1,8)], dim=0) # 9x8

            elif mode in ['train', 'val']:
                gamma1Trans = self.gamma1_processors[block](QG1,gamma1,'train') # train mode
                beta1Trans  = self.beta1_processors[block](QB1,beta1, 'train') # train mode
                gamma2Trans = self.gamma2_processors[block](QG2,gamma2,'train') # train mode
                beta2Trans  = self.beta2_processors[block](QB2,beta2, 'train') # train mode

            block_param_dict = {
                'gamma1': ( alpha*(gamma1Trans.view(-1)) + (1-alpha)*((gamma1De*gate).sum(1)) ) * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': ( alpha*(beta1Trans.view(-1)) + (1-alpha)*((beta1De*gate).sum(1)) ) * self.beta1_regularizers[block],
                'gamma2': ( alpha*(gamma2Trans.view(-1)) + (1-alpha)*((gamma2De*gate).sum(1)) ) * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': ( alpha*(beta2Trans.view(-1)) + (1-alpha)*((beta2De*gate).sum(1)) ) * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)

        if mode in ['train', 'val']:
            return block_params

        elif mode=='test':
            return block_params, alpha, softmaxTrans


    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term

