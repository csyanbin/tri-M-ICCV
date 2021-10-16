"""
    Configure Dict for Method details: 
        {model_idx:, model_name:, classifier, feature_adaptation_file:, feature_adaptation_module, classifier_adaptation_file, classifier_adaptation_module}
"""

MahSpecCoop_config  = {
                    'model_idx':                    0, 
                    'model_name':                   "MahSpecCoop", 
                    'feature_adaptation_file':      "adaptation_networks_MahSpecCoop", 
                    'feature_adaptation_module':    "FilmAdaptationNetwork", 
                    'feature_extractor_file':       'resnet'
                    #'classifier':                   "linear_classifier", # not used
                    #'classifier_adaptation_file':   "adaptation_networks_cos2", # not used  
                    #'classifier_adaptation_module': "LinearClassifierAdaptationNetwork"} # not used
                    }

# Candidate models 
choices = ["MahSpecCoop"]

# Dictionary mapping from model names to their config files
choices_dict = {'MahSpecCoop': MahSpecCoop_config}


## Return Domain Classification logits for loss computation
def ifreturn_logits(feature_adaptation):
    if feature_adaptation in  ["MahSpecCoop"]:
        return True
    else:
        return False


