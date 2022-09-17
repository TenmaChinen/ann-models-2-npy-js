import numpy as np

class Model:
    def __init__(self):
        self.l_layers = {str_code_l_layers_class_instantiation}
        
    def predict(self,x):
        output = x.copy()
        for layer in self.l_layers:
            output = layer(output)
            
        return output
    
{str_code_layer_classes}
{str_code_activation_functions}