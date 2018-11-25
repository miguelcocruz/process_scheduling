import os

def load_data(index):
    '''Returns a list where each element represents a line'''

    project_dir = os.path.join(os.path.curdir, os.pardir)
    raw_data_path = os.path.join(project_dir,'data','raw')
    filepath = os.path.join(raw_data_path, 'ejemplares_calibrado', 'ejemplar_calibrado_{}.txt'.format(index))

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