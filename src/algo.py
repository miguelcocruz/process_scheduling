import numpy as np
import os

def load_data(index):
    '''Returns a list where each element represents a line'''

    project_dir = os.path.join(os.path.curdir, os.pardir)
    raw_data_path = os.path.join(project_dir,'data','raw')
    filepath = os.path.join(raw_data_path, 'ejemplares_calibrado', 'ejemplar_calibrado_1.txt')

    lines = []

    with open(filepath, 'r') as file:
        for line in file:
            lines.append(line)

    return lines


def parse_data(lines):
    '''Returns parameters N, M, H, tin, tax, b and sec'''
    
    _n = int(lines[0].strip('\n'))
    _m = int(lines[1].strip('\n'))
    _h = int(lines[2].strip('\n'))

    _b = [int(_) for _ in lines[3 + 2 * _n].strip('\n').split('*')]

    _tin = []
    _tax = []
    _sec = []

    for i in range(3, 3 + _n):
        _tin.append([int(_) for _ in lines[i].strip('\n').split('*')]) 

    for i in range(3 + _n, 3 + 2 * _n):
        _tax.append([int(_) for _ in lines[i].strip('\n').split('*')])

    for i in range(3 + 2 * _n + 1, 3 + 3 * _n + 1):
        _sec.append([int(_) for _ in lines[i].strip('\n').split('*')])

    return _n, _m, _h, _b, _tin, _tax, _sec

def write_output(tb, ts, tf, fo, ti, index):
    '''Writes a textfile according to the required structure in the directory data/processed/sol_e.txt, where e represents the number of the dataset'''
    
    project_dir = os.path.join(os.path.curdir, os.pardir)
    proc_data_path = os.path.join(project_dir,'data','processed')
    filepath = os.path.join(proc_data_path, 'ejemplares_calibrado', 'ejemplar_calibrado_{}.txt'.format(index))

    with open(filepath, 'w') as file:
        
        for _f, _t in zip(fo, time):
            file.write("{}*{}\n".format(str(_f), str(_t))) #tirar str?
            
        file.write("{}\n".format(str(len(fo))))
        
        for _tb, _ts, _tf in zip(tb, ts, tf):
            file.write("{}*{}*{}\n".format("*".join(map(str,map(int,_tb))), \
                                          "*".join(map(str,map(int,_ts))), \
                                          str(int(_tf))))
def first_assignment(index):
    '''Assigns the request with a given index to be the first request. Returns tb, ts and tf'''
    global M, H, tin, sec
    
    _tb = np.zeros(shape=(M))
    _ts = np.zeros(shape=(H))
    _tin = np.array(tin[index])
    _sec = np.array(sec[index])
    
    for j in range(1, M):
        if j == 0:
            _tb[0] = 0
        
        else:
            _tb[j] = _tb[j - 1] + _tin[j - 1]
            
    for h, j in enumerate(b):
        _ts[h] = _tb[j]
        _tb[j:] += _sec[b.index(j)]
    
    _tf = _tb[M - 1] + _tin[M - 1]
    return _tb, _ts, _tf

def assign_new_request(tb_prev, ts_prev, tf_prev, index, compute_max):
    '''Assigns a new request based on a already assigned request.'''
    
    global M, sec, tin, tax
    
    _tb = np.zeros(shape=(M))
    _ts = np.zeros(shape=(H))
    _tin = np.array(tin[index])
    _sec = np.array(sec[index])
    _tax = np.array(tax[index])
    _b = np.array(b)
    
    for j in range(M):
        
        if j == 0:
            _tb[0] = tb_prev[1]
        
        elif j < M - 1:
            _tb[j] = max(_tb[j - 1] + _tin[j - 1], tb_prev[j+1])
        else:
            _tb[j] = max(_tb[j - 1] + _tin[j - 1], tf_prev)
     
    for h, j in enumerate(b):
        j -= 1
        
        if h == 0:
            _tb[j:] -= _tb[j] - ts_prev[h]
        elif h > 0:
            _tb[j:] -= min(_tb[j] - (_tb[j - 1] + _tin[j - 1]), _tb[j] - ts_prev[h], _tb[j] - _ts[h - 1] - _sec[h - 1])
        else:
            _tb[j:] -= min(_tb[j] - (_tb[j - 1] + _tin[j - 1]), _tb[j] - ts_prev[h])
            
        _ts[h] = _tb[j] + _tin[j]
        delta = tb_prev[j + 1:] - _tb[j:-1]

        if j + 1 in b:  
            _tb[j + 1:] += max(_sec[h], _sec[h] + ts_prev[b.index(j + 1)] - _tb[j + 1], *delta)
        else:
            _tb[j + 1:] += max(_sec[h], _sec[h] + tb_prev[j + 2] - _tb[j + 1], *delta)
   
    _tf = _tb[M - 1] + _tin[M - 1]
               
    if compute_max:
        for j in range(M):
            
            if j + 1 in b : 
                #doesn´t mean the next j, it is just because b doesn't start at zero         
                h = b.index(j + 1)
                
                if _ts[h] > _tax[j] + _tb[j]:
                    shift = _ts[h] - _tb[j] - _tax[j]
                    _tb[:j + 1] += shift
                    _ts[_b < j + 1] += shift
               
            elif j < M - 1 and _tb[j + 1] > _tax[j] + _tb[j]:
                shift = _tb[j + 1] - _tb[j] - _tax[j]
                _tb[:j + 1] += shift
                _ts[_b < j + 1] += shift
                
            elif j == M - 1 and tf_prev > _tb[j]:
                shift = tf_prev - _tb[j]
                _tb += shift
                _ts += shift
                _tf += shift
                
    return _tb, _ts, _tf