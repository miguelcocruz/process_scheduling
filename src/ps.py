import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

class Schedule():
    
    def load_data(self, type, index):

        raw_data_path = os.path.join(os.path.curdir,'data','raw')
        filepath = os.path.join(raw_data_path, 'ejemplares_{}'.format(type), 'ejemplar_{}_{}.txt'.format(type, index))

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

    def assign_new_request(self, tb_prev, ts_prev, tf_prev, index):
        '''Assigns a new request based on a already assigned request.'''

        _tb = np.zeros(shape=(self.M))
        _ts = np.zeros(shape=(self.H))
        _tin = np.array(self.tin[index])
        _sec = np.array(self.sec[index])
        _tax = np.array(self.tax[index])
        _b = np.array(self.b)

        _h = -1

        _jh = 0

        for j in range(self.M):

            if j == 0:
                
                if j + 1 in self.b:
                    _tb[0] = ts_prev[0]
                    _ts[0] = _tb[0] + _tin[0]
                    _h = 0
                    _jh = 1

                else:
                    _tb[0] = tb_prev[1]
                
            elif j < self.M - 1:
                
                if j in self.b:
                    _jh = j

                    if j + 1 in self.b:
                        _h += 1
                        _tb[j] = max(_ts[_h - 1] + _sec[_h - 1], ts_prev[_h])

                        _ts[_h] = _tb[j] + _tin[j]
                        
                    else:
                        _tb[j] = max(_ts[_h] + _sec[_h], tb_prev[j+1])

                else:
                    
                    if j + 1 in self.b:
                        _h += 1
                        _tb[j] = max(_tb[j - 1] + _tin[j - 1], ts_prev[_h])
                        _ts[_h] = _tb[j] + _tin[j]

                    else:
                        _tb[j] = max(_tb[j - 1] + _tin[j - 1], tb_prev[j+1])
                    
                    if _tb[j] - _tb[j - 1] > _tax[j - 1]:

                        #_tb[_jh:j] += _tb[j] - _tb[j - 1] - _tax[j - 1]

                        dif = _tb[j] - _tb[j - 1] - _tax[j - 1]

                        for k in reversed(range(_jh,j)):
                            #print(j, k)
                            if k in self.b or k == 0:
                                _tb[k:j] += dif
                                break

                            
                            else:
                                shift =  + _tb[k - 1] -_tb[k] + _tax[k - 1]
                                _tb[k:j] += min(dif, shift)
                                dif -= shift
                                if dif <= 0:
                                    break

                    
            else:
                
                if j in self.b:
                    _tb[j] = max(_ts[_h] + _sec[_h], tf_prev)
                
                else:
                    _tb[j] = max(_tb[j - 1] + _tin[j - 1], tf_prev)

                    if _tb[j] - _tb[j - 1] > _tax[j - 1]:
                        _tb[_jh:j] += _tb[j] - _tb[j - 1] - _tax[j - 1]

                        dif = _tb[j] - _tb[j - 1] - _tax[j - 1]

                        for k in reversed(range(_jh,j)):
                            if k in self.b or k == 0:
                                _tb[k:j] += dif
                                break
                            
                            else:
                                shift = -_tb[k] + _tb[k - 1] + _tax[k - 1]
                                _tb[k:j] += min(dif, shift)
                                dif -= shift
                                if dif <= 0:
                                    break


        _tf = _tb[self.M - 1] + _tin[self.M - 1]

        return _tb, _ts, _tf

    def compute_used_time(self, tb, ts, tf):
        '''Returns a list of tuples of the following form (starting time in bath i, time spent in bath i)'''

        aux = [(tb[j], tb[j+1] - tb[j]) if j + 1 not in self.b else (tb[j], ts[self.b.index(j + 1)] - tb[j]) for j in range(len(tb) - 1)]
        
        aux.append((tb[-1], tf - tb[-1]))

        return aux

    def compute_sec(self, tb, ts, tf, index):
        '''Returns a list of tuples of the following form (starting time at drying step h, time spent on drying step h)'''

        aux = [(ts[h], sec) for h, sec in zip(range(self.H), self.sec[index])]

        return aux

    def compute_min(self, tb, ts, tf, index):
        '''Returns a list of tuples of the following form (starting time in bath i, tin of bath i)'''

        aux = [(tb[j], tin) if j + 1 not in self.b else (tb[j], tin) for j, tin in zip(range(len(tb) - 1), self.tin[index])]

        aux.append((tb[-1], tf - tb[-1]))

        return aux

    def idle_time(self, tb1, ts1, tf1, tb2, ts2, tf2, j_start = None, j_end = None):
        '''Returns the sum of the idle time between two requests'''

        j_start = 0 if j_start is None else j_start
        j_end = self.N if j_start is None else j_start

        _dis = [abs(tb2[j] - tb1[j + 1]) if j + 1 not in self.b else abs(tb2[j] - ts1[self.b.index(j + 1)]) for j in range(self.M - 1)]
        _dis.append(abs(tf1 - tb2[-1]))

        return sum(_dis)

    def assign_all(self, first):
        
        S = Solution(self)

        tb_aux, ts_aux, tf_aux = self.first_assignment(first)

        unused_requests = set(range(self.N)) - {first}

        S.add_request(tb_aux, ts_aux, tf_aux, first)

        i_prev = first

        while len(unused_requests) > 0:

            new = True
            ind_opt = 0

            for i in unused_requests:

                tb2, ts2, tf2 = self.assign_new_request(S.TB[i_prev], S.TS[i_prev], S.TF[i_prev], i)

                ind = self.idle_time(S.TB[i_prev], S.TS[i_prev], S.TF[i_prev], tb2, ts2, tf2)

                if ind < ind_opt or new:
                    ind_opt = ind
                    tb_opt, ts_opt, tf_opt = tb2.copy(), ts2.copy(), tf2.copy()
                    i_add = i
                    new = False
            
            S.add_request(tb_opt, ts_opt, tf_opt, i_add)

            unused_requests -= {i_add}
            i_prev = i_add      
           
        return S

    def assign_new_set(self, S, i_request, i_prev, pos_h):
        
        tb_prev = S.TB[i_prev]
        ts_prev = S.TS[i_prev]
        tf_prev = S.TF[i_prev]

        tb = S.TB[i_request].copy()
        ts = S.TS[i_request].copy()
        tf = 0
        
        if pos_h == self.H:
            start = self.b[pos_h - 1]
            end = self.M
        
        elif pos_h == 0:
            start = 0
            end = self.b[pos_h] # don't subtract 1, so it is included in the range

        else:
            start = self.b[pos_h - 1]
            end = self.b[pos_h]
        
        for j in range(start, end):

            if j == 0:

                if j + 1 in self.b: # next to bath 0 is drying step h
                    tb[0] = ts_prev[0]
                    ts[0] = tb[0] + self.tin[i_request][0]

                else:
                    tb[0] = tb_prev[1]

            elif j < self.M - 1:

                if j in self.b: # bath j is preceded by a drying step

                    if j + 1 in self.b:

                        tb[j] = max(ts[pos_h - 1] + self.sec[i_request][pos_h - 1], ts_prev[pos_h])
                        ts[pos_h] = tb[j] + self.tin[i_request][j]

                    else:
                        tb[j] = max(ts[pos_h - 1] + self.sec[i_request][pos_h - 1], tb_prev[j + 1])

                else:

                    if j + 1 in self.b:                            
                        tb[j] = max(tb[j - 1] + self.tin[i_request][j - 1], ts_prev[pos_h])
                        ts[pos_h] = tb[j] + self.tin[i_request][j]

                    else:
                        tb[j] = max(tb[j - 1] + self.tin[i_request][j - 1], tb_prev[j + 1])

                    if tb[j] - tb[j - 1] > self.tax[i_request][j - 1]:

                        dif = tb[j] - tb[j - 1] - self.tax[i_request][j - 1]

                        for k in reversed(range(start, j)):
                            if k in self.b or k == 0:
                                tb[k:j] += dif
                                break
                            else:
                                shift = tb[k - 1] - tb[k] + self.tax[i_request][k - 1]
                                tb[k:j] += min(dif, shift)
                                dif -= shift
                                if dif <= 0:
                                    break


            else:

                if j in self.b:
                    tb[j] = max(ts[pos_h - 1] + self.sec[i_request][pos_h - 1], tf_prev)

                else:
                    tb[j] = max(tb[j - 1] + self.tin[i_request][j - 1], tf_prev)

                    if tb[j] - tb[j - 1] > self.tax[i_request][j - 1]:

                        dif = tb[j] - tb[j - 1] - self.tax[i_request][j - 1]

                        for k in reversed(range(start, j)):
                            if k in self.b or k == 0:
                                tb[k:j] += dif
                                break
                            else:
                                shift = tb[k - 1] - tb[k] + self.tax[i_request][k - 1]
                                tb[k:j] += min(dif, shift)
                                dif -= shift
                                if dif <= 0:
                                    break

            if pos_h == self.H:
                tf = tb[self.M - 1] + self.tin[i_request][self.M - 1]

        return tb, ts, tf

    def assign_by_order(self, order, i_max = None):

        i_max = self.N if i_max is None else i_max

        S = Solution(self)
        
        i_prev = 0 #does not matter since all times equal 0 in the begining
        
        for h in range(self.H + 1):
            
            for i in order[:i_max, h]:
                
                tb, ts, tf = self.assign_new_set(S, i, i_prev, h)
                S.add_request(tb, ts, tf, i, h, 0, i_max)

                i_prev = S.Order[S.current_pos - 1, h]
    
        return S

    def assign_by_order_2(self, solution, order, f_opt, i_min = 0, h_min = 0):

        S = solution.copy()
        
        S.current_pos = i_min

        for h in range(h_min, self.H + 1):
            i_prev = order[i_min - 1, h]
            S.current_pos = i_min

            for i in order[i_min:, h]:
                tb, ts, tf = self.assign_new_set(S, i, i_prev, h)
                
                S.add_request(tb, ts, tf, i, h)

                if S.TF[i] > f_opt:
                    return S
                
                i_prev = i
                
    
        return S

    def local_search(self, solution, direction = 'backward', first_improving = False, search_range = 5):
    
        S = solution.copy()
        SX = Solution(self)
        
        f_opt = max(S.TF)

        if direction == 'forward':
            search_order = range(self.N)
        
        elif direction == 'backward':
            search_order = list(reversed(range(self.N)))
        
        for h in range(self.H + 1):

            for pos_1 in search_order:
                change_found = False
                search_limits = range(self.N) if h == 0 else range(max(pos_1 - search_range, 0), min(pos_1 + search_range, self.N))

                for pos_2 in search_limits:
                    
                    order = np.delete(S.Order[:, h:], pos_1, axis=0)
                    order = np.insert(order, pos_2, S.Order[pos_1, h:], axis=0)
                    order = np.concatenate((S.Order[:, :h], order), axis=1)
                    
                    if pos_1 == 0 or pos_2 == 0:
                        SX = self.assign_by_order(order)

                    else:
                        SX = self.assign_by_order_2(S, order, f_opt, min(pos_1, pos_2), h)

                    f = np.max(SX.TF)

                    if f < f_opt:
                        f_opt = f
                        pos_2_opt = pos_2
                        change_found = True

                        if first_improving:
                            break

                if change_found:
                    order_opt = np.delete(S.Order[:, h:], pos_1, axis=0)
                    order_opt = np.insert(order_opt, pos_2_opt, S.Order[pos_1, h:], axis=0)
                    order_opt = np.concatenate((S.Order[:, :h], order_opt), axis=1)

                    S = self.assign_by_order(order_opt)
 
        return S

    def greedy_constructor(self, method, weights, alpha):
        S = Solution(self)
        S.empty()

        S0 = Solution(self)
        S0.empty()
       

        if method == 'min_tin':
            total_tin = np.array([sum(tin) for tin in self.tin]) #from smallest to largest

            order = list(total_tin.argsort())

        elif method == 'max_tin':
            total_tin = np.array([sum(tin) for tin in self.tin]) #from smallest to largest

            order = list(reversed(total_tin.argsort()))

        elif method == 'min_tin_sec':
            total = np.array([sum(tin) + sum(sec) for tin, sec in zip(self.tin, self.sec)])

            order = list(total.argsort())

        elif method == 'max_tin_sec':
            total = np.array([sum(tin) + sum(sec) for tin, sec in zip(self.tin, self.sec)])

            order = list(reversed(total.argsort()))   
        
        first = order[0]

        tb, ts, tf = self.first_assignment(first)

        S.add_request(tb, ts, tf, first)

        count = -1

        for i in order[1:]:
            count += 1
            nr_comb = 0

            S1 = []
            result = dict()
            
            while nr_comb < count + 2:
                S0.Order = np.insert(S.Order, nr_comb, [i] * (self.H + 1), axis=0)
                S1.append(self.assign_by_order(S0.Order, count + 2))

                result[nr_comb] = weights[0] * np.max(S1[-1].TF) + weights[1] * np.mean(S1[-1].TB)

                S0.Order = np.delete(S0.Order, -1, axis=0)

                nr_comb += 1
            
            c_min = min(result.values())
            c_max = max(result.values())
            
            RCL = [i for i in result.keys() if result[i] <= c_min + alpha * (c_max - c_min)]
            
            ind = random.sample(RCL, 1)[0]
            
            S = S1[ind].copy()
            
        SX = self.assign_by_order(S.Order)
        
        return SX

    def greedy_constructor_2(self, method, weights, alpha, first_improving = False, search_range = 5):
        S = Solution(self)
        S.empty()

        S0 = Solution(self)
        S0.empty()
       

        if method == 'min_tin':
            total_tin = np.array([sum(tin) for tin in self.tin]) #from smallest to largest

            order = list(total_tin.argsort())

        elif method == 'max_tin':
            total_tin = np.array([sum(tin) for tin in self.tin]) #from smallest to largest

            order = list(reversed(total_tin.argsort()))

        elif method == 'min_tin_sec':
            total = np.array([sum(tin) + sum(sec) for tin, sec in zip(self.tin, self.sec)])

            order = list(total.argsort())

        elif method == 'max_tin_sec':
            total = np.array([sum(tin) + sum(sec) for tin, sec in zip(self.tin, self.sec)])

            order = list(reversed(total.argsort()))   
        
        first = order[0]

        tb, ts, tf = self.first_assignment(first)

        S.add_request(tb, ts, tf, first)

        for e, i in enumerate(order[1:]):

            new_it = True
            change_found = True
            #print('position_e', e, 'req_i', i)

            while change_found:

                change_found = False

                for h in range(self.H + 1):
                    search_limits = range(e + 1) if h == 0 else range(max(pos - search_range, 0), min(e + 1, pos + search_range, self.N))

                    for pos in search_limits:
                        
                        if h == 0:
                            order = np.insert(S.Order, pos, [i] * (self.H + 1),  axis = 0)
                            #print(order)

                        else:
                            order = np.delete(S.Order[:, h:], pos_prev, axis = 0)
                            order = np.insert(order, pos, S.Order[pos_prev, h:], axis = 0)
                            order = np.concatenate((S.Order[:, :h], order), axis = 1)
                    
                        SX = self.assign_by_order(order, e + 1)
                        
                        f = np.mean(SX.TF)
                        #print('h', h, 'pos', pos, 'cost', f)

                        if new_it or f < f_opt:
                            f_opt = f
                            new_it = False
                            change_found = True
                            order_opt = order.copy()
                            pos_prev = pos

                            #print('h', h, 'cost', f)

                            if first_improving:
                                break

                    if change_found:

                        S = self.assign_by_order(order_opt)
                        #print(order_opt[:e + 3,:])
                        
                        #print(order_opt[:e + 3,:])

        return S

                        







        SX = self.assign_by_order(S.Order)
        
        return SX
    
    def plot_schedule(self, solution, order_column = 0):
        fig, ax = plt.subplots()
        co = ['lightcoral', 'orange', 'plum', 'grey', 'royalblue', 'maroon', 'coral', 'g', 'red', 'sienna']

        pp = - 0.9

        ax.set_ylim(0, self.N)

        for i in solution.Order[:, order_column]:
            pp += 1
            ax.broken_barh(self.compute_min(solution.TB[i], solution.TS[i], solution.TF[i], i), yrange=(pp,0.8), color=co)
            ax.broken_barh(self.compute_used_time(solution.TB[i], solution.TS[i], solution.TF[i]), yrange=(pp+0.3,0.3), color=co)
            ax.broken_barh(self.compute_sec(solution.TB[i], solution.TS[i], solution.TF[i], i), yrange=(pp+0.35,0.2), color='lightblue')
            ax.text(solution.TB[i][0] - 100, pp, i)

def write_output(tb, ts, tf, fo, ti, type, index):
    
    '''Writes a textfile according to the required structure in the directory data/processed/sol_e.txt, where e represents the number of the dataset'''
    
    proc_data_path = os.path.join(os.path.curdir,'data','processed')
    filepath = os.path.join(proc_data_path, 'ejemplares_{}'.format(type), 'so_{}.txt'.format(index))

    with open(filepath, 'w') as file:
        
        for _f, _t in zip(fo, ti):
            file.write("{}*{}\n".format(str(_f), str(_t))) #tirar str?
            
        file.write("{}\n".format(str(len(fo))))

        file.write("{}*{}\n".format(str(_f), str(_t)))
        
        for _tb, _ts, _tf in zip(tb, ts, tf):
            file.write("{}*{}*{}\n".format("*".join(map(str,map(int,_tb))), \
                                        "*".join(map(str,map(int,_ts))), \
                                        str(int(_tf))))

def get_benchmark(type, index):
    project_dir = os.path.join(os.path.curdir, os.pardir)
    raw_data_path = os.path.join(project_dir,'data','raw')
    filepath = os.path.join(raw_data_path, "@valores_ejemplares_{}.txt".format(type))

    solution_lines = []

    with open(filepath, 'r') as file:
        for line in file:
            solution_lines.append(line)

    solution_value = int(solution_lines[index+1].strip().split()[-1])

    return solution_value

class Solution():

    def __init__(self, Schedule, empty = True):
        self.Schedule = Schedule
        
        if empty:
            self.empty()
    

    def define(self, TB, TS, TF, Order, Last):
        self.TB = TB.copy()
        self.TS = TS.copy()
        self.TF = TF.copy()
        self.Order = Order.copy()

        self.current_pos = Last
        
    def empty(self):
        '''Creates the time variables and the Order matrix'''

        self.TB = [np.zeros((self.Schedule.M)) for _ in range(self.Schedule.N)]
        self.TS = [np.zeros((self.Schedule.H)) for _ in range(self.Schedule.N)]
        self.TF = [0 for _ in range(self.Schedule.N)]
        self.Order = np.full(shape=(self.Schedule.N, self.Schedule.H + 1), fill_value=-1, dtype='int')

        self.current_pos = 0

    def add_request(self, tb, ts, tf, index_request, h = None, reset_position = None, when = None):
        '''Add to the solution the times of a given request.'''

        self.TB[index_request] = tb.copy()
        self.TS[index_request] = ts.copy()
        self.TF[index_request] = tf

        if h is None:
            self.Order[self.current_pos, :] = index_request
            
        else:
            self.Order[self.current_pos, h] = index_request

        when = self.Schedule.N if when is None else when

        if self.current_pos == when - 1:
            self.current_pos = self.current_pos if reset_position is None else reset_position

        else:
            self.current_pos += 1

    def copy(self):
        S_copy = Solution(self.Schedule)
        S_copy.TB = copy.deepcopy(self.TB)
        S_copy.TS = copy.deepcopy(self.TS)
        S_copy.TF = copy.deepcopy(self.TF)
        S_copy.Order = copy.deepcopy(self.Order)
        S_copy.current_pos = self.current_pos

        return S_copy


    









