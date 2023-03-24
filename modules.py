from sklearn.utils import class_weight
import numpy as np


def Add_class_wight(y):#inout=pd.series ->  output=weight(np.aaray) 
    class_weights = list(class_weight.compute_class_weight('balanced', 
                                                           classes=np.unique(y),
                                                           y=y)
                        )
    w_array = np.ones(y.shape[0], dtype = 'float16')
    for i, val in enumerate(y):
        w_array[i] = class_weights[int(val)]
        
    return w_array
    