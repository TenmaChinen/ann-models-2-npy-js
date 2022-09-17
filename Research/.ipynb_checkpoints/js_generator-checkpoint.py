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
    
    l_w, l_b, l_a = get_weights_and_acts(l_layers_cnf,d_model_weights)
    h5_file.close()
    return l_w, l_b, l_a


def get_layers_cnf(d_model_config):
    l_keys = ['name','units','dtype','activation','use_bias']
    l_layers_cnf = []
    for layer in d_model_config['config']['layers']:
        if layer['class_name'] not in ['InputLayer','Dropout' ] :
            l_layers_cnf.append({ k : layer['config'][k] for k in l_keys })
    return l_layers_cnf
    

def load_model_config(json_path):
    file = open(json_path,'r')
    d_model_config = json.load(file)
    file.close()
    return d_model_config


def get_weights_and_acts(l_layers_cnf,d_model_weights):
    l_w, l_b, l_a = [], [], []

    for layer_cnf in l_layers_cnf:
        name = layer_cnf['name']
        activation = layer_cnf['activation']
        d_weights = d_model_weights[name][name]

        kernel = d_weights['kernel:0'][:]
        bias = d_weights['bias:0'][:] if layer_cnf['use_bias'] else 0

        l_w.append(kernel)
        l_b.append(bias)
        l_a.append(activation)
        
    return l_w, l_b, l_a

#################################################################################
######################   P Y T H O N   G E N E R A T O R   ######################
#################################################################################

def get_py_code(l_w,l_b,l_a):
    str_code = get_py_class_template()
    str_code = str_code.format(
        act_fncs = get_py_act_fnc(l_a),
        l_w = str(list(map(lambda w : w.tolist(), l_w))),
        l_b = str(list(map(lambda b : b if type(b) is int else b.tolist(), l_b))),
        l_a  = str(l_a)
    )
    return str_code

def get_py_code_activations():
    return load_code_str('py_activations.py')

def get_py_class_template():
    return load_code_str('py_template.py')

def get_py_act_fnc(l_activations):
    str_code_act = get_py_code_activations()
    string = ''
    for activation in set(l_activations):
        pattern = f'(def )?{activation}[^;]*'
        result = re.search(pattern,str_code_act)
        if result:
            string += result.group() + '\n'
            
    string += '\nd_act = { ' + ', '.join([ f"'{act}' : {act}" for act in set(l_activations)]) + ' }\n\n'
    return string

#################################################################################
##################   J A V A S C R I P T   G E N E R A T O R   ##################
#################################################################################

def get_js_code(l_w,l_b,l_a):
    str_code = get_js_class_template()
    
    str_code = re.sub('(?<!#){', '#[' ,str_code)
    str_code = re.sub('}(?!#)', ']#', str_code)
    
    str_code = str_code.replace('#{','{')
    str_code = str_code.replace('}#','}')
    
    str_code = str_code.format(
        act_fncs = get_js_act_fnc(l_a),
        l_w = '[ ' + ','.join(map(parse_weight, l_w)) + ']',
        l_b = '[ ' + ','.join(map(parse_bias, l_b)) + ']',
        l_a  = str(l_a)
    )
    
    str_code = str_code.replace('#[','{')
    str_code = str_code.replace(']#','}')
    
    str_code += "\n\n" + get_js_matrix()
    return str_code

def parse_weight(w):
    return f'new Matrix({str(w.tolist())})'

def parse_bias(b):
    return str(b) if type(b) is int else f'new Matrix([{str(b.tolist())}])' 

def get_js_class_template():
    return load_code_str('js_template.js')

def get_js_code_activations():
    return load_code_str('js_activations.js')

def get_js_act_fnc(l_activations):
    str_code_act = get_js_code_activations()
    string = re.search('(?<=#)function applyFunc[^#]*', str_code_act).group() + '\n\n'
    for activation in set(l_activations):
        pattern = f'(?<=#){activation}[^#]*'
        result = re.search(pattern,str_code_act)
        if result:
            string += result.group() + '\n'
            
    string += 'dActs = { ' + ', '.join([ f"'{act}' : {act}" for act in set(l_activations)]) + ' }'
    
    return string

def get_js_matrix():
    return load_code_str('js_matrix.js')

#################################################################################

def load_code_str(file_name):
    file = open(file_name,'r')
    str_code = file.read()
    file.close()
    return str_code