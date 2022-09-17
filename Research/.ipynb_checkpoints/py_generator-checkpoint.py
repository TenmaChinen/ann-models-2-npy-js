import numpy as np
import h5py
import json
import re

#################################################################################
###################   M O D E L   F I L E S   H A N D L E R   ###################
#################################################################################

def get_model_data(h5_path,json_path=None):
    '''
        output: l_layers_cnf
    '''
    h5_file = h5py.File(h5_path,'r')

    if h5_file.attrs['keras_version'] != '2.7.0' :
        print('Keras Version == 2.7.0 Only Supported')
        return None

    if 'model_config' in h5_file.attrs:
        d_model_config = json.loads(h5_file.attrs['model_config'])
        d_model_weights = h5_file['model_weights']
    elif json_path:
        d_model_config = load_model_config(json_path)
        d_model_weights = h5_file
    else:
        print('Json File Needed!')
        return None
    
    if d_model_config['class_name'] != 'Sequential' :
        print('Non Sequential Models Still Not Supported')
        return None
    
    l_layers_cnf = get_layers_cnf(d_model_config)
    l_layers_objs = get_weights_and_acts(l_layers_cnf,d_model_weights)
    
    h5_file.close()
    return l_layers_objs


def get_layers_cnf(d_model_config):
    l_keys = ['name','units','dtype','activation','use_bias']
    l_layers_cnf = []
    for layer in d_model_config['config']['layers']:
        if layer['class_name'] not in [ 'InputLayer','Dropout' ] :
            d_layer_cnf = { k : layer['config'][k] for k in l_keys }
            d_layer_cnf[ 'class_name' ] = layer['class_name']
            l_layers_cnf.append(d_layer_cnf)
    return l_layers_cnf


def get_weights_and_acts(l_layers_cnf,d_model_weights):
    l_layers_obj = []
    
    for d_layer_cnf in l_layers_cnf:

        class_name = d_layer_cnf['class_name']
        layer_name = d_layer_cnf['name']
        use_bias = d_layer_cnf['use_bias']
        activation = d_layer_cnf['activation']
        d_layer_data = { 'type' : class_name }
        
        if class_name == 'Dense':
            d_weights = d_model_weights[layer_name][layer_name]
            kernel = d_weights['kernel:0'][:]
            bias = d_weights['bias:0'][:] if use_bias else None
            d_layer_data.update( {'activation' : activation, 'weights' : [kernel,bias] } )
            
        elif class_name == 'LSTM':
            d_weights, *_ = d_model_weights[layer_name][layer_name].values()
            units = d_layer_cnf['units']
            kernel = d_weights['kernel:0'][:]
            bias = d_weights['bias:0'][:]
            rec_kernel = d_weights['recurrent_kernel:0'][:]
            d_layer_data.update( {'activation' : activation, 'units' : units, 'weights' : [kernel,bias,rec_kernel]} )
        
        l_layers_obj.append( d_layer_data )
        
    return l_layers_obj

def load_model_config(json_path):
    file = open(json_path,'r')
    d_model_config = json.load(file)
    file.close()
    return d_model_config

#################################################################################
######################   P Y T H O N   G E N E R A T O R   ######################
#################################################################################

def get_model_code(l_layers_obj):
    psc = PyStringCode()
    
    l_layer_types = { layer['type'] for layer in l_layers_obj }
    l_layer_acts = { layer['activation'] for layer in l_layers_obj if layer.get('activation') }
    
    str_code_layer_classes = psc.get_code_layers_class(l_layer_types)
    str_code_act_funcs = psc.get_code_activations_func(l_layer_acts)
    
    l_str_code_layers_instant = []
    
    for d_layer_obj in l_layers_obj:
        layer_type = d_layer_obj['type']
        d_kwargs = { k : d_layer_obj[k] for k in ['activation','units'] if d_layer_obj.get(k) }
        
        if 'weights' in d_layer_obj:
            d_kwargs['l_weights'] = list_to_str(d_layer_obj['weights'])
    
        str_args = ', '.join([ f'{k}={v}' for k,v in d_kwargs.items()])
        l_str_code_layers_instant.append( f'{layer_type}({str_args})' )
     
    str_code_l_layers_classes_instant = list_to_str(l_str_code_layers_instant)
    
    str_code_model_template = psc.get_code_model_template()
    str_code_model = str_code_model_template.format(
        str_code_l_layers_class_instantiation = str_code_l_layers_classes_instant,
        str_code_layer_classes = str_code_layer_classes,
        str_code_activation_functions = str_code_act_funcs
    )
        
    return str_code_model

def array_to_str(array):
    return f'np.array({array.tolist()})'

def list_to_str(l_list):
    l_list = [ array_to_str(item) if is_array(item) else item for item in l_list ]
    return f"[ {', '.join(l_list)} ]"


def is_array(item):
    return type(item) is np.ndarray

class PyStringCode:
    
    def __init__( self ):
        self.str_layers = self.get_code_layers()
        self.str_code_act = self.get_code_activations()
        
    def get_code_layer_class(self, layer_type):
        pattern = f'class {layer_type}[^;]*'
        result = re.search( pattern, self.str_layers)
        if result : return result.group()
    
    def get_code_layers_class(self,l_layers_type):
        return '\n'.join(map(self.get_code_layer_class, l_layers_type))
    
    def get_code_activation_func(self, activation_name):
        pattern = f'(def )?{activation_name}[^;]*'
        result = re.search(pattern, self.str_code_act)
        if result:
            return result.group()
        return ''
            
    def get_code_activations_func(self, l_activations):
        return '\n'.join(map(self.get_code_activation_func, l_activations))

    @staticmethod
    def get_code_model_template():
        return load_code_str('py_template.py')

    @staticmethod
    def get_code_activations():
        return load_code_str('py_activations.py')

    @staticmethod
    def get_code_layers():
        return load_code_str('py_layers.py')
    
def load_code_str(file_name):
    file = open(file_name,'r')
    str_code = file.read()
    file.close()
    return str_code

#################################################################################

