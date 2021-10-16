from resnet import film_resnet18, resnet18
from set_encoder import SetEncoder

import conf_dict
from importlib import import_module

""" Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
"""


class ConfigureNetworks:
    def __init__(self, pretrained_resnet_path, feature_adaptation):
        
        config_dict = conf_dict.choices_dict
        config = config_dict[feature_adaptation] # config for method

        # print model configs
        print('Model config:')
        for k,v in config.items():
            print('\t', k, ':', v)
        
        ## ------------- <<Classifier: Mahalanobis Distance>> ----------- ##
        # self.classifier = getattr(import_module('utils'), config['classifier'])
        # print(self.classifier)

        ## ------------- <<Set Encoder>> ---------- ##
        self.encoder = SetEncoder()
        z_g_dim = self.encoder.pre_pooling_fn.output_size

        # parameters for ResNet18
        num_maps_per_layer = [64, 128, 256, 512]
        num_blocks_per_layer = [2, 2, 2, 2]
        num_initial_conv_maps = 64
        
        ## --------------- <<feature adaptation network>> -----------  ##
        print("feature adaptation:", feature_adaptation)
        adaptation_networks_module = import_module(config['feature_adaptation_file'])
        FilmAdaptationNetwork = getattr(adaptation_networks_module, config['feature_adaptation_module'])    
        FilmLayerNetwork = getattr(adaptation_networks_module, 'FilmLayerNetwork')
        
        # deal with dynamic resnet files
        resnet_file = config.get('feature_extractor_file', 'resnet')
        resnet = import_module(resnet_file) # import resnet
        film_resnet18 = getattr(resnet, 'film_resnet18')
        self.feature_extractor = film_resnet18(
            pretrained=True,
            pretrained_model_path=pretrained_resnet_path
        )
        self.feature_adaptation_network = FilmAdaptationNetwork(
            layer=FilmLayerNetwork,
            num_maps_per_layer=num_maps_per_layer,
            num_blocks_per_layer=num_blocks_per_layer,
            z_g_dim=z_g_dim
        )

        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        ## ----------------- <<classifier adaptation network>> --------------- ##
        if feature_adaptation.find('Mah')>=0:
            self.classifier_adaptation_network = None
            print('Mahalanobis classifier')
        else:
            classifier_adaptation_file = import_module(config['classifier_adaptation_file'])
            LinearClassifierAdaptationNetwork = getattr(classifier_adaptation_file, config['classifier_adaptation_module'])
            self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.output_size)

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_classifier_adaptation(self):
        return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
