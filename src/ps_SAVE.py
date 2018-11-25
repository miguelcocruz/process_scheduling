import os
import numpy as np

class Schedule():
    
    def load_data(self, index):
        '''Returns a list where each element represents a line'''

        project_dir = os.path.join(os.path.curdir, os.pardir)
        raw_data_path = os.path.join(project_dir,'data','raw')
        filepath = os.path.join(raw_data_path, 'ejemplares_prueba', 'ejemplar_prueba_{}.txt'.format(index))

        lines = []

        with open(filepath, 'r') as file:
            for line in file:
                lines.append(line)

        self.N = int(lines[0].strip('\n'))
        self.M = int(lines[1].strip('\n'))
        self.H = int(lines[2].strip('\n'))

        self.b = [int(_) for _ in lines[3 + 2 * self.N].strip('\n').split('*')]

        self.tin = []
        self.tax = []
        self.sec = []

        for i in range(3, 3 + self.N):
            self.tin.append([int(_) for _ in lines[i].strip('\n').split('*')]) 

        for i in range(3 + self.N, 3 + 2 * self.N):
            self.tax.append([int(_) for _ in lines[i].strip('\n').split('*')])

        for i in range(3 + 2 * self.N + 1, 3 + 3 * self.N + 1):
            self.sec.append([int(_) for _ in lines[i].strip('\n').split('*')])


    def first_assignment(self, index):
        '''Assigns the request with a given index to be the first request. Returns tb, ts and tf'''
        
        _tb = np.zeros(shape=(self.M))
        _ts = np.zeros(shape=(self.H))
        _tin = np.array(self.tin[index])
        _sec = np.array(self.sec[index])
        
        for j in range(1, self.M):
            if j == 0:
                _tb[0] = 0
            
            else:
                _tb[j] = _tb[j - 1] + _tin[j - 1]
                
        for h, j in enumerate(self.b):
            _ts[h] = _tb[j]
            _tb[j:] += _sec[self.b.index(j)]
        
        _tf = _tb[self.M - 1] + _tin[self.M - 1]
        return _tb, _ts, _tf

    def assign_new_request_og(self, tb_prev, ts_prev, tf_prev, index, compute_max):
        '''Assigns a new request based on a already assigned request.'''

        _tb = np.zeros(shape=(self.M))
        _ts = np.zeros(shape=(self.H))
        _tin = np.array(self.tin[index])
        _sec = np.array(self.sec[index])
        _tax = np.array(self.tax[index])
        _b = np.array(self.b)

        for j in range(self.M):
            
            if j == 0:
                _tb[0] = tb_prev[1]
            
            elif j < self.M - 1:
                _tb[j] = max(_tb[j - 1] + _tin[j - 1], tb_prev[j+1])
            else:
                _tb[j] = max(_tb[j - 1] + _tin[j - 1], tf_prev)
            
        for h, j in enumerate(self.b):
            j -= 1
            
            if j == 0:
                _tb[j:] -= _tb[j] - ts_prev[h]
            elif h > 0:
                _tb[j:] -= min(_tb[j] - (_tb[j - 1] + _tin[j - 1]), _tb[j] - ts_prev[h], _tb[j] - _ts[h - 1] - _sec[h - 1])
            else:
                _tb[j:] -= min(_tb[j] - (_tb[j - 1] + _tin[j - 1]), _tb[j] - ts_prev[h])
                
            _ts[h] = _tb[j] + _tin[j]
            delta = tb_prev[j + 1:] - _tb[j:-1]

            if j + 1 in self.b:  
                _tb[j + 1:] += max(_sec[h], _sec[h] + ts_prev[self.b.index(j + 1)] - _tb[j + 1], *delta)
            else:
                _tb[j + 1:] += max(_sec[h], _sec[h] + tb_prev[j + 2] - _tb[j + 1], *delta)

        _tf = _tb[self.M - 1] + _tin[self.M - 1]
        
        new_j = 0

        if compute_max:
            for j in range(self.M):
                
                if j + 1 in self.b : 
                    #doesn´t mean the next j, it is just because b doesn't start at zero         
                    h = self.b.index(j + 1)
                    
                    if _ts[h] > _tax[j] + _tb[j]:
                        shift = _ts[h] - _tb[j] - _tax[j]
                        _tb[new_j:j + 1] += shift
                        _ts[h] += shift ##sub h por _b < j + 1
                        print(j, shift) #NÃO OCORRE?
                    
                    new_j = j + 1

                elif j < self.M - 1 and _tb[j + 1] > _tax[j] + _tb[j]:
                    shift = _tb[j + 1] - _tb[j] - _tax[j]
                    _tb[new_j:j + 1] += shift
                    
                elif j == self.M - 1 and tf_prev > _tb[j]:
                    shift = tf_prev - _tb[j]
                    _tb[new_j:] += shift
                    _tf += shift
                    print(j, shift) #NÃO OCORRE?
                    
        return _tb, _ts, _tf

    def assign_new_request(self, tb_prev, ts_prev, tf_prev, index, compute_max):
        '''Assigns a new request based on a already assigned request.'''

        _tb = np.zeros(shape=(self.M))
        _ts = np.zeros(shape=(self.H))
        _tin = np.array(self.tin[index])
        _sec = np.array(self.sec[index])
        _tax = np.array(self.tax[index])
        _b = np.array(self.b)

        last_h = -1
    
        for j in range(self.M):

            if j == 0:
                _tb[0] = tb_prev[1]
                
                if j + 1 in self.b:
                    _ts[0] = _tb[0] + _tin[0]
                    last_h = 0
                
            elif j < self.M - 1:
                
                if j in self.b:
                    
                    if j + 1 in self.b:
                        last_h += 1
                        _tb[j] = max(_ts[last_h - 1] + _sec[last_h - 1], ts_prev[last_h])
                        _ts[last_h] = _tb[j] + _tin[j]
                        
                    else:
                        _tb[j] = max(_ts[last_h] + _sec[last_h], tb_prev[j+1])
                
                else:
                    
                    if j + 1 in self.b:
                        last_h += 1
                        _tb[j] = max(_tb[j - 1] + _tin[j - 1], ts_prev[last_h])
                        _ts[last_h] = _tb[j] + _tin[j]
                    
                    else:
                        _tb[j] = max(_tb[j - 1] + _tin[j - 1], tb_prev[j+1])
                    
            else:
                
                if j in self.b:
                    _tb[j] = max(_ts[last_h] + _sec[last_h], tf_prev)
                
                else:
                    _tb[j] = max(_tb[j - 1] + _tin[j - 1], tf_prev)

        _tf = _tb[self.M - 1] + _tin[self.M - 1]
    
        
        new_j = 0

        if compute_max:
            for j in range(self.M):
                
                if j + 1 in self.b : 
                    #doesn´t mean the next j, it is just because b doesn't start at zero         
                    h = self.b.index(j + 1)
                    
                    if _ts[h] > _tax[j] + _tb[j]:
                        shift = _ts[h] - _tb[j] - _tax[j]
                        _tb[new_j:j + 1] += shift
                        _ts[h] += shift ##sub h por _b < j + 1
                        print(j, shift) #NÃO OCORRE?
                    
                    new_j = j + 1

                elif j < self.M - 1 and _tb[j + 1] > _tax[j] + _tb[j]:
                    shift = _tb[j + 1] - _tb[j] - _tax[j]
                    _tb[new_j:j + 1] += shift
                    
                elif j == self.M - 1 and tf_prev > _tb[j]:
                    shift = tf_prev - _tb[j]
                    _tb[new_j:] += shift
                    _tf += shift
                    print(j, shift) #NÃO OCORRE?
                    
        return _tb, _ts, _tf


    def compute_Vb(self, tb, ts, tf):
        aux = [(tb[j], tb[j+1] - tb[j]) if j + 1 not in self.b else (tb[j], ts[self.b.index(j + 1)] - tb[j]) for j in range(len(tb) - 1)]
        
        aux.append((tb[-1], tf - tb[-1]))

        return aux

    def dissimilarity(self, tb1, ts1, tf1, tb2, ts2, tf2):
        _v1 = self.compute_Vb(tb1, ts1, tf1)
        _v2 = self.compute_Vb(tb2, ts2, tf2)
    
        _dis = [abs(sum(req_1) - req_2[0]) for req_1, req_2 in zip(_v1[:self.M], _v2[1:])]

        return sum(_dis)

    def dissimilarity_2(self, tb1, ts1, tf1, tb2, ts2, tf2):
        _v1 = self.compute_Vb(tb1, ts1, tf1)
        _v2 = self.compute_Vb(tb2, ts2, tf2)
    
        _dis = [(sum(req_1) - sum(req_2)) for req_1, req_2 in zip(_v1[:self.M], _v2[1:])]

        return sum(_dis)

    def above_min(self, tb2, ts2, tf2, index):
        _v = self.compute_Vb(tb2, ts2, tf2)

        aux_b = [v[1] - t for v, t in zip(_v, self.tin[index])]
    
        aux_h = [tb2[self.b[h]] - ts2[h] for h, j in enumerate(ts2)]
        
        return sum(aux_b) + sum(aux_h)

    def assign_all(self, first, method):
        
        tb1, ts1, tf1 = self.first_assignment(first)
        _tb = [None for _ in range(self.N)]
        _ts = [None for _ in range(self.N)]
        _tf = [None for _ in range(self.N)]
        tb_aux, ts_aux, tf_aux = [], [], []

        _tb[first], _ts[first], _tf[first] = tb1.copy(), ts1.copy(), tf1.copy()

        unused_requests = set(range(self.N)) - {first}

        order = [first]
        while len(unused_requests) > 0:
            ds = []
            new = True
            ds_opt = 0

            for i in unused_requests:
                #tb2, ts2, tf2 = self.assign_new_request_og(tb1, ts1, tf1, i, compute_max=True)
                tb2, ts2, tf2 = self.assign_new_request(tb1, ts1, tf1, i, compute_max=True)

                #if tf2 < tf2_1:
                ds.append(self.dissimilarity(tb1, ts1, tf1, tb2, ts2, tf2) / (tf2 / np.mean(self.tax[i])))
                #else:
                #    ds.append(self.dissimilarity(tb1, ts1, tf1, tb2_1, ts2_1, tf2_1))

                if ds[-1] < ds_opt or new:
                    ds_opt = ds[-1].copy()
                    tb_opt, ts_opt, tf_opt = tb2.copy(), ts2.copy(), tf2.copy()
                    i_add = i
                    new = False

            new = True
            unused_requests -= {i_add}
            order.append(i_add)

            _tb[i_add] = tb_opt.copy()
            _ts[i_add] = ts_opt.copy()
            _tf[i_add] = tf_opt.copy()

            tb1, ts1, tf1 = tb_opt.copy(), ts_opt.copy(), tf_opt.copy()
        
        return _tb, _ts, _tf, order

    def permute_order(self, tb, ts, tf, order, h_perm, index):
    
        j_perm = self.b[h_perm] - 1
        
        ind2 = order[index]
        ind1 = order[index - 1]

        _tb = [_.copy() for _ in tb]
        _ts = [_.copy() for _ in ts]
        _tf = [_ for _ in tf]

        # times of ind2
        
        mat_fin = np.matrix(_tb) + np.matrix(self.compute_T(_tb, _ts, _tf))
        
        submat_fin = mat_fin[order[:index-1],:]    
        
        for j in range(j_perm + 1, self.M):
            
            if j in self.b: #if j is preceded by a drying step
                #tempo minimo de secado e pedidos anteriores
                _tb[ind2][j] = max(_ts[ind2][self.b.index(j)] + self.sec[ind2][self.b.index(j)], int(max(submat_fin[:,j])))
            
            else:
                _tb[ind2][j] = max(_tb[ind2][j - 1] + self.tin[ind2][j - 1], int(max(submat_fin[:,j])))
                
            if j + 1 in self.b:
                _ts[ind2][self.b.index(j + 1)] = _tb[ind2][j] + self.tin[ind2][j]

            elif j == self.M - 1:
                _tf[ind2] = _tb[ind2][j] + self.tin[ind2][j]
            
        # falta corrigir tax
        
        new_j = j_perm + 1
        
        for j in range(j_perm + 1, self.M):
            
            if j + 1 in self.b:
                #if this bath is followed by a drying step
                new_j = j + 1
            
            elif j < self.M - 1 and _tb[ind2][j + 1] > _tb[ind2][j] + self.tax[ind2][j]:
                _tb[ind2][new_j:j + 1] += _tb[ind2][j + 1] - _tb[ind2][j] - self.tax[ind2][j]
                
        
        # times of ind1
        
        for j in range(j_perm + 1, self.M):
            
            if j in self.b: #if j is preceded by a drying step
                #tempo minimo de secado e pedidos anteriores
                if j + 1 in self.b:

                    _tb[ind1][j] = max(_ts[ind1][self.b.index(j)] + self.sec[ind1][self.b.index(j)], int(max(submat_fin[:,j])), _ts[ind2][self.b.index(j + 1)])
                    
                elif j < self.M - 1:
                    _tb[ind1][j] = max(_ts[ind1][self.b.index(j)] + self.sec[ind1][self.b.index(j)], int(max(submat_fin[:,j])), _tb[ind2][j + 1])
                
                else:
                    _tb[ind1][j] = max(_ts[ind1][self.b.index(j)] + self.sec[ind1][self.b.index(j)], int(max(submat_fin[:,j])), _tf[ind2])

            elif j < self.M - 1:
                if j + 1 in self.b:  
                    _tb[ind1][j] = max(_tb[ind1][j - 1] + self.tin[ind1][j - 1], int(max(submat_fin[:,j])), _ts[ind2][self.b.index(j + 1)])
                    
                else:
                    _tb[ind1][j] = max(_tb[ind1][j - 1] + self.tin[ind1][j - 1], int(max(submat_fin[:,j])), _tb[ind2][j + 1])
                    
            else:
                _tb[ind1][j] = max(_tb[ind1][j - 1] + self.tin[ind1][j - 1], int(max(submat_fin[:,j])), _tf[ind2])
                
            if j + 1 in self.b:
                _ts[ind1][self.b.index(j + 1)] = _tb[ind1][j] + self.tin[ind1][j]

            elif j == self.M - 1:
                _tf[ind1] = _tb[ind1][j] + self.tin[ind1][j]
        
        # tax for ind1
        
        new_j = j_perm + 1
        
        for j in range(j_perm + 1, self.M):
            
            if j + 1 in self.b:
                #if this bath is followed by a drying step
                new_j = j + 1
            
            elif j < self.M - 1 and _tb[ind1][j + 1] > _tb[ind1][j] + self.tax[ind1][j]:
                _tb[ind1][new_j:j + 1] += _tb[ind1][j + 1] - _tb[ind1][j] - self.tax[ind1][j]
                
        # falta reajustar seguintes pedidos
        
        
        for i, ind in enumerate(order[index + 1:]):
            
            mat_fin = np.matrix(_tb) + np.matrix(self.compute_T(_tb, _ts, _tf))
            
            submat_fin = mat_fin[order[:index + i + 1],:]

            for j in range(j_perm + 1, self.M):
                
                if j in self.b: #if j is preceded by a drying step
                    #tempo minimo de secado e pedidos anteriores
                    _tb[ind][j] = max(_ts[ind][self.b.index(j)] + self.sec[ind][self.b.index(j)], int(max(submat_fin[:,j])))

                else:
                    _tb[ind][j] = max(_tb[ind][j - 1] + self.tin[ind][j - 1], int(max(submat_fin[:,j])))

                if j + 1 in self.b:
                    _ts[ind][self.b.index(j + 1)] = _tb[ind][j] + self.tin[ind][j]

                elif j == self.M - 1:
                    _tf[ind] = _tb[ind][j] + self.tin[ind][j]
        
            new_j = j_perm + 1
            
            for j in range(j_perm + 1, self.M):

                if j + 1 in self.b:
                    #if this bath is followed by a drying step
                    new_j = j + 1

                elif j < self.M - 1 and _tb[ind][j + 1] > _tb[ind][j] + self.tax[ind][j]:
                    _tb[ind][new_j:j + 1] += _tb[ind][j + 1] - _tb[ind][j] - self.tax[ind][j]
                
                
        return _tb, _ts, _tf

    def compute_T(self, tb, ts, tf):
        '''returns a vector T_ij whose elements represent the time the request i spent on bath j'''
    
    
        T = np.zeros(shape=(self.N, self.M))
        
        for i in range(self.N):

            for j in range(self.M):
                if j == self.M - 1:
                    T[i, j] = tf[i] - tb[i][j]
                elif j + 1 in self.b:
                    T[i, j] = ts[i][self.b.index(j+1)] - tb[i][j]
                else:
                    T[i, j] = tb[i][j+1] - tb[i][j]
                
        return T






def write_output(tb, ts, tf, fo, ti, index):
    
    '''Writes a textfile according to the required structure in the directory data/processed/sol_e.txt, where e represents the number of the dataset'''
    
    project_dir = os.path.join(os.path.curdir, os.pardir)
    proc_data_path = os.path.join(project_dir,'data','processed')
    filepath = os.path.join(proc_data_path, 'ejemplares_prueba', 'so_{}.txt'.format(index))

    with open(filepath, 'w') as file:
        
        for _f, _t in zip(fo, ti):
            file.write("{}*{}\n".format(str(_f), str(_t))) #tirar str?
            
        file.write("{}\n".format(str(len(fo))))

        file.write("{}*{}\n".format(str(_f), str(_t)))
        
        for _tb, _ts, _tf in zip(tb, ts, tf):
            file.write("{}*{}*{}\n".format("*".join(map(str,map(int,_tb))), \
                                        "*".join(map(str,map(int,_ts))), \
                                        str(int(_tf))))
