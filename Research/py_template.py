import numpy as np

{act_fncs}

class Model:
    def __init__(self):
        self.l_w = {l_w}
        self.l_b = {l_b}
        self.l_a = {l_a}
        
    def pred(self,x):
        output = x.copy()
        for w,b,act in zip(l_w,l_b,l_a):
            f_act = d_act[act]
            output = f_act(output.dot(w) + b)
        return output