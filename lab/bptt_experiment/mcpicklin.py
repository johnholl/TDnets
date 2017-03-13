import pickle
import numpy as np
from scipy import stats

def load_obj(name ):
    with open('/home/john/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

a = load_obj(name='MC_experiment_200000')

print(stats.mode(a.values()))
print(np.average(a.values()))
