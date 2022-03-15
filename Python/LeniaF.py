import math, numpy as np                        # (pip for macbook) pip3 install numpy
import scipy.ndimage                            # pip3 install scipy
import scipy.spatial.distance, scipy.signal
import reikna.fft, reikna.cluda                 # pip3 install pyopencl/pycuda, reikna
import skimage.morphology, skimage.segmentation # pip3 install scikit-image
import skimage._shared.coord
import PIL.Image, PIL.ImageTk                   # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
try: import tkinter as tk
except: import Tkinter as tk
from fractions import Fraction
import copy, re, itertools, json, csv
import io, os, sys, argparse, datetime, time, string, subprocess, multiprocessing
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from scipy.ndimage.zoom()
warnings.filterwarnings('ignore', '.*divide by zero encountered.*')  # suppress warning from divide by zero in kernel_core()
warnings.filterwarnings('ignore', '.*overflow encountered in exp.*')  # suppress warning from exp in kernel_core()
warnings.filterwarnings('ignore', '.*invalid value encountered.*')  # suppress warning from divide by zero in normalize(), get_stat_row()
warnings.filterwarnings('ignore', '.*nperseg.*') # suppress warning from scipy.signal.periodogram/welch
#python -c "import numpy; print(numpy.__version__)"

'''
JSON versions
major changes in from_data(), kernel_shell(), calc_kernel(), calc_once()
v3.0 Lenia.py = original (e.g. animals.json)
    {code, name, cname, params{R,T,b,m,s,kn,gn}, cells"$"}
v3.3 LeniaND.py = multi-dimensional (add cell delimiters)
    {code, name, cname, params{R,T,b,m,s,kn,gn}, cells"$%#"}
v3.4 LeniaNDK.py = multi-kernel (multiple params, e.g. 213.json)
    {code, name, cname, params[{R,T,b,m,s,kn,gn}], cells"$%#"}
v3.5.0 LeniaNDKC.py = multi-channel (multiple cells, e.g. 233.json)
    {code, name, cname, params[{R,T,b,m,s,kn,gn,h,r,c[]}], cells["$%#"]}
v3.5.1 (add settings for alternatives [arita, clip], e.g. 233tt1.json)
    {code, name, cname, settings{}, params[{R,T,b,m,s,kn,gn,h,r,c[]}], cells["$%#"]}
v3.5.2 (add model)
    {code, name, cname, settings{}, model{R,T,P,kn,gn} params[{b,m,s,h,r,c[]}], cells["$%#"]}
v3.6.0 LeniaF.py = free kernel (add free_h, adjust values T and h)
    {code, name, cname, settings{}, model{R,T*,P,kn,gn} params[{b,m,s,h*,r,c[]}], cells["$%#"]}
v3.6.1 (add free_b, divide b into rings, separate c0 c1, e.g. 233o.json)
    {code, name, cname, settings{}, model{R,T,P,kn,gn} params[{m,s,h,c0,c1,rings[{r,w,b}]}], cells["$%#"]}
'''

is_free_h = True
is_free_b = False

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, 
description='''Lenia in n-Dimensions    by Bert Chan 2020

recommended settings: (2D) -d2 -p2, (wide) -d2 -p0 -w 10 9, (3D) -d3 -p3, (4D) -d4 -p4''')
parser.add_argument('-g', '--gpu', dest='G', action='store_true', help='interactive choose GPU')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-s', '--size', dest='S', default=None, action='store', type=int, nargs='+', help='exact array size (apply to all sides if only one value, default 2^(W-P))')
group.add_argument('-w', '--win', dest='W', default=[9], action='store', type=int, nargs='+', help='window size = 2^W (apply to all sides if only one value, default 2^9 = 512)')
group.add_argument('--wide', action="store_true", help='wide window (i.e. -w 10 9)')
parser.add_argument('-p', '--pixel', dest='P', default=None, action='store', type=int, help='pixel size = 2^P (default 2^D)')
parser.add_argument('-b', '--border', dest='B', default=0, action='store', type=int, help='pixel border (default 0)')
parser.add_argument('-d', '--dim', dest='D', default=2, action='store', type=int, help='number of dimensions (default 2)')
parser.add_argument('-c', '--channel', dest='C', default=1, action='store', type=int, help='number of channels (default 1)')
parser.add_argument('-k', '--kernel', dest='K', default=1, action='store', type=int, help='number of self-connecting kernels (default 1)')
parser.add_argument('-x', '--cross', dest='X', default=1, action='store', type=int, help='number of cross-connecting kernels (default 1)')
parser.add_argument('-f', '--found', dest='F', default=None, action='store', type=str, help='found animals filename (default <DCK>.json)')
args = parser.parse_args()

# W,W,P,B   GoL 9,9,3,1   Lenia Lo 9,9,2,0  Hi 9,9,0,0   1<<7=128x128

DIM = args.D
DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}
X_AXIS, Y_AXIS, Z_AXIS = -1, -2, -3

PIXEL_2 = args.P if args.P is not None else args.D
PIXEL_BORDER = args.B
if args.S is None:
    if args.wide:
        SIZE_2 = [win_2 - PIXEL_2 for win_2 in (10, 9)]
    else:
        SIZE_2 = [win_2 - PIXEL_2 for win_2 in args.W]
    if len(SIZE_2) < DIM:
        SIZE_2 += [SIZE_2[-1]] * (DIM-len(SIZE_2))
    SIZE = [1 << size_2 for size_2 in SIZE_2]
else:
    SIZE = args.S  #160
    if len(SIZE) < DIM:
        SIZE += [SIZE[-1]] * (DIM-len(SIZE))
    SIZE_2 = [math.log(size,2) for size in SIZE]

#size    pixel mac
#128x128 16384 26.7  -w 9 9
#140x140 25600 20.4
#160x160 25600 17.7
#180x180 32400 15.0
#256x128 32768 14.2  -w 10 9
#192x192 36864 13.0
#200x200 40000 12.2

PIXEL = 1 << PIXEL_2
MID = [int(size / 2) for size in SIZE]
SIZEX, SIZEY = SIZE[0], SIZE[1]
MIDX, MIDY = MID[0], MID[1]

SIZER, SIZETH, SIZEF = min(MIDX, MIDY), SIZEX, MIDX
DEF_R   = int(np.power(2.0, min(SIZE_2) - 6) * DIM * 5)  # default 20
RAND_R1 = int(np.power(2.0, min(SIZE_2) - 7) * DIM * 5)  # default 10
RAND_R2 = int(np.power(2.0, min(SIZE_2) - 5) * DIM * 5)  # default 40

CN = args.C
KN = args.K
XN = args.X
CHANNEL = range(CN)
KERNEL = range(KN*CN + XN*CN*(CN-1))

ALIVE_THRESHOLD = 0.1
EPSILON = 1e-10
ROUND = 10
DEFAULT_RING = {'r':0.5, 'w':0.5, 'b':1}
EMPTY_RING = {'r':0.5, 'w':0.5, 'b':0}
STATUS = []
is_windows = (os.name == 'nt')
np.set_printoptions(precision=3)

class Board:
    def __init__(self, size=[0]*DIM):
        self.names = {'code':'', 'name':'', 'cname':''}
        self.settings = {}
        self.model = {'R':DEF_R, 'T':10, 'P':0, 'kn':1, 'gn':1};
        #self.params = [{'b':[1], 'm':0.1, 's':0.01, 'h':1, 'r':1} for k in KERNEL]
        self.params = [{'rings':[DEFAULT_RING.copy()], 'm':0.1, 's':0.01, 'h':1, 'c0':0, 'c1':0} for k in KERNEL]
        self.cells = [np.zeros(size) for c in CHANNEL]

    @classmethod
    def from_values(cls, cells):
        self = cls()
        self.cells = copy.deepcopy(cells) if cells is not None else None
        return self

    def init_channels(self):
        i = 0
        for c0 in CHANNEL:
            for k in range(KN):
                p = self.params[i]
                if 'c0' not in p:
                    p['c0'] = c0
                    p['c1'] = c0
                i += 1
        for c0 in CHANNEL:
            for c1 in CHANNEL:
                if c0 != c1:
                    for k in range(XN):
                        p = self.params[i]
                        if 'c0' not in p:
                            p['c0'] = c0
                            p['c1'] = c1
                        i += 1

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = {'code':data.get('code',''), 'name':data.get('name',''), 'cname':data.get('cname','')}
        self.settings = data.get('settings', None)
        self.model = data.get('model', None)
        params = data.get('params')
        if params is not None:
            if type(params) not in [list]:
                params = [params for k in KERNEL]
            self.params = [Board.data2params(p) for p in params]
            if 'c0' not in self.params[0]:  # compatibility before v3.6.1 free_b
                if 'c' in self.params[0]:
                    for p in self.params:
                       p['c0'], p['c1'] = p.pop('c')
                else:
                    self.init_channels()
            if self.model is None:  # compatibility before v3.5.2 add model
                self.model = {}
                for k, default in zip(('R', 'T', 'P', 'kn', 'gn'), (DEF_R, 10, 0, 1, 1)):
                    self.model[k] = self.params[0].get(k, default)
                for p in self.params:
                    for k in ('R', 'T', 'kn', 'gn'):
                        p.pop(k, None)
                if is_free_h: self.free_h()  # compatibility before v3.6.0 free_h
            if is_free_b: self.free_b()  # compatibility before v3.6.1 free_b
            if self.settings is None:  # compatibility before v3.5.1 add settings
                self.settings = {}
        self.cells = None
        rle = data.get('cells')
        if rle is not None:
            if type(rle) not in [list]:
                rle = [rle for c in CHANNEL]
            self.cells = [Board.rle2cells(r) for r in rle]
            for c in range(CN - len(self.cells)):
                self.split_channel(len(self.cells)-1)  #(c)
            #self.split_channel(0)
            #self.split_channel(2)
            #self.split_channel(3)
        #print(self.model)
        #print(repr(self.params).replace("}, {", "},\r\n{"))
        return self

    def to_data(self, is_shorten=True):
        rle = [Board.cells2rle(self.cells[c], is_shorten) for c in CHANNEL]
        params = [Board.params2data(self.params[k]) for k in KERNEL]
        data = {'code':self.names['code'], 'name':self.names['name'], 'cname':self.names['cname'], 'settings':self.settings, 'model':self.model, 'params':params, 'cells':rle}
        return data

    def free_h(self):
        # h -> h / sum(h over same c1)
        Dn = [0 for c in CHANNEL]
        max_h = 0
        for p in self.params:
            Dn[p['c1']] += p['h']
        for p in self.params:
            p['h'] = round(p['h'] / Dn[p['c1']], 3)
            max_h = max(max_h, p['h'])
        if max_h <= 1/2:
            for p in self.params:
                p['h'] *= 2
            self.model['T'] *= 2

    def free_b(self):
        for p in self.params:
            r = p.pop('r')
            if 'rings' not in p and 'b' in p:
                b = p.pop('b')
                B2 = 2 * len(b)
                p['rings'] = [ {
                    'r':round(r*(2*i+1)/B2, 2), 
                    'w':round(r/B2, 2), 
                    'b':round(float(b_i), 2)
                } for i,b_i in enumerate(b) ]

    def params2st(self, params=None, is_brief=False):
        if params is not None:
            params2 = Board.params2data(params, add_brackets=True)
            if 'rings' in params2:
                return ','.join(["b={v}".format(v=[ring['b'] for ring in v]) if k=='rings' else "{k}={v}".format(k=k,v=str(v)) for (k,v) in params2.items()])  #if k not in ('kn', 'gn')
            else:
                return ','.join(["{k}={v}".format(k=k,v=str(v)) for (k,v) in params2.items()])  #if k not in ('kn', 'gn')
        else:
            st = ['{' + self.params2st(self.params[k]) + '}' for k in KERNEL]
            return ', '.join(st)

    def long_name(self):
        # return ' | '.join(filter(None, self.names))
        return "{code} | {name} {cname}".format(**self.names)

    @staticmethod
    def ch2val(c):
        if c in '.b': return 0
        elif c == 'o': return 255
        elif len(c) == 1: return ord(c)-ord('A')+1
        else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)

    @staticmethod
    def val2ch(v):
        if v == 0: return ' .'
        elif v < 25: return ' ' + chr(ord('A')+v-1)
        else: return chr(ord('p') + (v-25)//24) + chr(ord('A') + (v-25)%24)

    @staticmethod
    def _recur_drill_list(dim, lists, row_func):
        if dim < DIM-1:
            return [Board._recur_drill_list(dim+1, e, row_func) for e in lists]
        else:
            return row_func(lists)

    @staticmethod
    def _recur_join_st(dim, lists, row_func):
        if dim < DIM-1:
            return DIM_DELIM[DIM-1-dim].join(Board._recur_join_st(dim+1, e, row_func) for e in lists)
        else:
            return DIM_DELIM[DIM-1-dim].join(row_func(lists))

    @staticmethod
    def _append_stack(list1, list2, count, is_repeat=False):
        list1.append(list2)
        if count != '':
            repeated = list2 if is_repeat else []
            list1.extend([repeated] * (int(count)-1))

    @staticmethod
    def _recur_get_max_lens(dim, list1, max_lens):
        max_lens[dim] = max(max_lens[dim], len(list1))
        if dim < DIM-1:
            for list2 in list1:
                Board._recur_get_max_lens(dim+1, list2, max_lens)

    @staticmethod
    def _recur_cubify(dim, list1, max_lens):
        more = max_lens[dim] - len(list1)
        if dim < DIM-1:
            list1.extend([[]] * more)
            for list2 in list1:
                Board._recur_cubify(dim+1, list2, max_lens)
        else:
            list1.extend([0] * more)

    @staticmethod
    def cells2rle(A, is_shorten=True):
        values = np.rint(A*255).astype(int).tolist()  # [[255 255] [255 0]]
        if is_shorten:
            rle_groups = Board._recur_drill_list(0, values, lambda row: [( len(list(g)), Board.val2ch(v).strip() ) for v,g in itertools.groupby(row)] )
            st = Board._recur_join_st(0, rle_groups, lambda row: [(str(n) if n>1 else '')+c for n,c in row] )  # "2 yO $ 1 yO"
        else:
            st = Board._recur_join_st(0, values, lambda row: [Board.val2ch(v) for v in row] )
        return st + '!'

    @staticmethod
    def rle2cells(st):
        stacks = [[] for dim in range(DIM)]
        last, count = '', ''
        delims = list(DIM_DELIM.values())
        st = st.rstrip('!') + DIM_DELIM[DIM-1]
        for ch in st:
            if ch.isdigit(): count += ch
            elif ch in 'pqrstuvwxy@': last = ch
            else:
                if last+ch not in delims:
                    Board._append_stack(stacks[0], Board.ch2val(last+ch)/255, count, is_repeat=True)
                else:
                    dim = delims.index(last+ch)
                    for d in range(dim):
                        Board._append_stack(stacks[d+1], stacks[d], count, is_repeat=False)
                        stacks[d] = []
                    #print("{0}[{1}] {2}".format(last+ch, count, [np.asarray(s).shape for s in stacks]))
                last, count = '', ''
        A = stacks[DIM-1]
        max_lens = [0 for dim in range(DIM)]
        Board._recur_get_max_lens(0, A, max_lens)
        Board._recur_cubify(0, A, max_lens)
        return np.asarray(A)

    @staticmethod
    def fracs2st(B):
        # [Fraction(1),Fraction(1/2)] -> "1,1/2"
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        # "1,1/2" -> [Fraction(1),Fraction(1/2)]
        return [Fraction(st) for st in st.split(',')]

    @staticmethod
    def params2data(p, add_brackets=False):
        p2 = p.copy()
        if 'b' in p2:
            p2['b'] = Board.fracs2st(p2['b'])
            if add_brackets:
               p2['b'] = '[' + p2['b'] + ']'
        return p2

    @staticmethod
    def data2params(p):
        p2 = p.copy()
        if 'b' in p2:
            p2['b'] = Board.st2fracs(p2['b'])
        p2.setdefault('h', 1)
        p2.setdefault('r', 1)
        return p2

    def clear(self):
        for c in CHANNEL:
            self.cells[c].fill(0)

    def _recur_add(self, dim, cells1, cells2, shift, is_centered, vmin):
        size1, size2 = cells1.shape[0], cells2.shape[0]
        size0 = min(size1, size2)
        start1 = (size1 - size0)//2 + shift[dim] if is_centered else shift[dim]
        start2 = (size2 - size0)//2 if is_centered else 0
        if dim < DIM-1:
            for x in range(size0):
                self._recur_add(dim+1, cells1[(start1+x)%size1], cells2[start2+x], shift, is_centered, vmin)
        else:
            for x in range(size0):
                if cells2[start2+x] > vmin:
                    cells1[(start1+x)%size1] = cells2[start2+x]

    def add(self, part, shift=[0]*DIM, is_centered=True):
        # shift: s, z, y, x
        # assert self.params['R'] == part.params['R']
        if type(shift[0]) not in [list]:
            shift = [shift for c in CHANNEL]
        vmin = part.model.get('vmin', EPSILON)
        for c in CHANNEL:
            self._recur_add(0, self.cells[c], part.cells[c], shift[c], is_centered, vmin)
        return self

    def transform(self, tx, mode='RZSF', z_axis=Z_AXIS, is_world=False):
        if 'R' in mode and tx['rotate'] != [0]*3:
            for c in CHANNEL:
                if DIM == 2:
                    self.cells[c] = scipy.ndimage.rotate(self.cells[c], -tx['rotate'][1], reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')
                elif DIM >= 3:
                    self.cells[c] = scipy.ndimage.rotate(self.cells[c], tx['rotate'][2], axes=(X_AXIS, z_axis), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by y axis = x-z plane
                    self.cells[c] = scipy.ndimage.rotate(self.cells[c], tx['rotate'][1], axes=(z_axis, Y_AXIS), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by x axis = z-y plane
                    self.cells[c] = scipy.ndimage.rotate(self.cells[c], tx['rotate'][0], axes=(Y_AXIS, X_AXIS), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by z axis = y-x plane
        if 'Z' in mode and tx['R'] != self.model['R']:
            # print("* {} / {}".format(tx['R'], self.params['R']))
            shape_orig = self.cells[0].shape
            for c in CHANNEL:
                self.cells[c] = scipy.ndimage.zoom(self.cells[c], tx['R'] / self.model['R'], order=0)
            if is_world:
                self.cells = Board(shape_orig).add(self).cells
            self.model['R'] = tx['R']
        if 'F' in mode and tx['flip'] != -1:
            extra_slice = [slice(None)] * (DIM-2)
            for c in CHANNEL:
                if tx['flip'] in [0,1]: self.cells[c] = np.flip(self.cells[c], axis=DIM-1-tx['flip'])
                elif tx['flip'] == 2: slice1 = [slice(None), slice(None,-MIDX-1,-1)]; slice2 = [slice(None), slice(None,MIDX)]; self.cells[c][tuple(extra_slice + slice1)] = self.cells[c][tuple(extra_slice + slice2)]
                elif tx['flip'] == 3: slice1 = [slice(None), slice(None,-MIDX-1,-1)]; slice2 = [slice(None,None,-1), slice(None,MIDX)]; self.cells[c][tuple(extra_slice + slice1)] = self.cells[c][tuple(extra_slice + slice2)]
                # elif tx['flip'] == 4: i_upper = np.triu_indices(SIZEX, -1); self.cells[i_upper] = self.cells.T[i_upper]
                elif tx['flip'] == 4: self.cells[c][tuple(extra_slice + [slice(None), slice(MIDX,None)])] = 0
                elif tx['flip'] == 5: self.cells[c][tuple(extra_slice + [slice(MIDY,None), slice(None)])] = 0
                elif tx['flip'] == 6: 
                    self.cells[c][tuple(extra_slice + [slice(None,MIDY//2), slice(None)])] = 0
                    self.cells[c][tuple(extra_slice + [slice(MIDY+MIDY//2,None), slice(None)])] = 0
                    self.cells[c][tuple(extra_slice + [slice(None), slice(None,MIDX//2)])] = 0
                    self.cells[c][tuple(extra_slice + [slice(None), slice(MIDX+MIDX//2,None)])] = 0
        if 'S' in mode and tx['shift'] != [0]*DIM:
            for c in CHANNEL:
                self.cells[c] = scipy.ndimage.shift(self.cells[c], tx['shift'], order=0, mode='wrap')
                # self.cells = np.roll(self.cells, tx['shift'], (2, 1, 0))
        return self

    def add_transformed(self, part, tx):
        part = copy.deepcopy(part)
        self.add(part.transform(tx, mode='RZF'), tx['shift'])
        return self

    def crop(self):
        #vmin = np.amin(self.cells)
        coords_list = [np.argwhere(self.cells[c] > ALIVE_THRESHOLD) for c in CHANNEL]
        coords = np.concatenate(coords_list)
        if coords.size == 0:
            self.cells = [np.zeros([1]*DIM) for c in CHANNEL]
        else:
            min_point = coords.min(axis=0)
            max_point = coords.max(axis=0) + 1
            slices = [slice(x1, x2) for x1, x2 in zip(min_point, max_point)]
            for c in CHANNEL:
                self.cells[c] = self.cells[c][tuple(slices)]
        return self

    def restore_to(self, dest):
        dest.models = copy.deepcopy(self.models)
        dest.params = copy.deepcopy(self.params)
        dest.cells = copy.deepcopy(self.cells)
        dest.names = self.names.copy()

    def copy_kernel(self, p, src=None, dest=None):
        new_p = copy.deepcopy(p)
        if src is not None:  new_p['c0'] = src
        if dest is not None: new_p['c1'] = dest
        self.params.append(new_p)
        return new_p

    def split_kernel(self, p, src=None, dest=None, new_h_ratio=1/2):
        h = p['h']
        new_p = copy.deepcopy(p)
        if src is not None:  new_p['c0'] = src
        if dest is not None: new_p['c1'] = dest
        p['h'] = (1-new_h_ratio) * h
        new_p['h'] = new_h_ratio * h
        self.params.append(new_p)
        return new_p

    '''
    for xn: RR -> AA/ BB/ AB/ BA/
    for kn-xn: RR -> AA BB
    XR -> XA XB
    RX -> AX/ BX/
    '''
    def split_channel(self, old_ch):
        new_ch = len(self.cells)
        self.cells.append(copy.deepcopy(self.cells[old_ch]))
        self_split_count = 0
        #print(str(self.params).replace('}, {', '},\r\n{'))
        for k in range(len(self.params)):
            p = self.params[k]
            c0, c1 = p.get('c0', 0), p.get('c1', 0)
            if c0==old_ch and c1==old_ch:
                p2 = self.copy_kernel(p, src=new_ch, dest=new_ch)
                if self_split_count<XN:
                    self.split_kernel(p, dest=new_ch)
                    self.split_kernel(p2, dest=old_ch)
                    self_split_count += 1
            elif c0!=old_ch and c1==old_ch:
                self.copy_kernel(p, dest=new_ch)
            elif c0==old_ch and c1!=old_ch:
                self.split_kernel(p, src=new_ch)
        #print(str(self.params).replace('}, {', '},\r\n{'))

class Automaton:
    kernel_core = {
        # [0,1] -> [0,1]
        1: lambda r: (r>0)*(r<1) * (4 * r * (1-r))**4,  # polynomial (quad4)
        2: lambda r: (r>0)*(r<1) * np.nan_to_num( np.exp( 4 - 1 / (r * (1-r)) ), 0),  # exponential / gaussian bump (bump4)
        3: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        4: lambda r: (r>0)*(r<1) * np.exp(- ((r-0.5)/0.15)**2 / 2)  # exponential / leaky gaussian bump
    }
    growth_func = {
        # [0,1] -> [-1,1]
        1: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        2: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        3: lambda n, m, s: (np.abs(n-m)<=s) * 2 - 1  # step (stpz)
    }

    def __init__(self, world):
        self.world = world
        self.world_FFT = [np.zeros(world.cells[0].shape) for c in CHANNEL]
        self.potential_FFT = [np.zeros(world.cells[0].shape) for k in KERNEL]
        self.potential = [np.zeros(world.cells[0].shape) for k in KERNEL]
        self.field = [np.zeros(world.cells[0].shape) for k in KERNEL]
        self.change = [np.zeros(world.cells[0].shape) for c in CHANNEL]
        self.X = [None]*DIM
        self.D = None
        self.Z_depth = None
        self.TH = None
        self.R = None
        self.polar_X = None
        self.polar_Y = None
        self.gen = 0
        self.time = 0
        self.soft_clip_level = 0
        self.world.model['vmin'] = EPSILON if self.soft_clip_level == 0 else ALIVE_THRESHOLD
        self.is_arita_mode = False
        self.arita_layers = []
        self.mask_rate = 0
        self.add_noise = 0
        self.is_inverted = False
        self.is_gpu = False
        self.has_gpu = True
        self.compile_gpu(self.world.cells[0])
        self.calc_kernel()

    def kernel_shell(self, R, model, params):
        kfunc = Automaton.kernel_core[model.get('kn')]
        if 'rings' in params:
            v = [ kfunc( (R - ring['r']) / (2*ring['w']) + 1/2 ) * ring['b'] for ring in params['rings'] ]
            return sum(v)
        elif 'b' in params:
            r = params['r']
            B = len(params['b'])
            Br = B * R / r
            bs = np.asarray([float(f) for f in params['b']])
            b = bs[np.minimum(np.floor(Br).astype(int), B-1)]
            return (R<r) * kfunc(np.minimum(Br % 1, 1)) * b

    @staticmethod
    def soft_max(x, m, k):
        ''' Soft maximum: https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
        return np.log(np.exp(k*x) + np.exp(k*m)) / k

    def soft_clip(self, x, min, max):
        #https://medium.com/life-at-hopper/clip-it-clip-it-good-1f1bf711b291
        if self.soft_clip_level == 1:
            #return np.tanh(2*x-1) / 2 + 0.5
            return 1 / (1 + np.exp(-4*x+2))
        else:
            k = np.exp(13-self.soft_clip_level)
            return -np.log(1/(np.power(k, x) + 1) + 1/k) / np.log(k)
        # a = np.exp(k*x)
        # b = np.exp(k*min)
        # c = np.exp(-k*max)
        # return np.log( 1/(a+b)+c ) / -k
        #return Automaton.soft_max(Automaton.soft_max(x, min, k), max, -k)

    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft1 = self.gpu_fftn = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create(interactive=args.G)
            self.gpu_fft1 = reikna.fft.FFT(A.astype(np.complex64),axes=[0]).compile(self.gpu_thr)
            self.gpu_fftn = reikna.fft.FFT(A.astype(np.complex64)).compile(self.gpu_thr)
            self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
        except Exception as e:
            # if str(e) == "No supported GPGPU APIs found":
            self.has_gpu = False
            self.is_gpu = False
            print(e)
            # raise e

    def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
        if self.is_gpu and self.gpu_thr and gpu_func:
            op_dev = self.gpu_thr.to_device(A.astype(dtype))
            gpu_func(op_dev, op_dev, **kwargs)
            return op_dev.get()
        else:
            return cpu_func(A)
            # return np.roll(potential_shifted, (MIDX, MIDY), (1, 0))

    #def fft1(self, A): return self.run_gpu(A, np.fft.fft, self.gpu_fft1, np.complex64)
    def fft1(self, A): return np.fft.fft(A)
    def fftn(self, A): return self.run_gpu(A, np.fft.fftn, self.gpu_fftn, np.complex64)
    def ifftn(self, A): return self.run_gpu(A, np.fft.ifftn, self.gpu_fftn, np.complex64, inverse=True)
    # def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)
    def fftshift(self, A): return np.fft.fftshift(A)

    def calc_once(self, is_update=True):
        A = self.world.cells
        R, T, P = [self.world.model[k] for k in ('R', 'T', 'P')]
        dt = 1 / T
        gfunc = Automaton.growth_func[self.world.model.get('gn')]
        self.world_FFT = [self.fftn(A[c]) for c in CHANNEL]
        D = [np.zeros(A[c].shape) for c in CHANNEL]
        if not is_free_h: Dn = [0 for c in CHANNEL]
        for k in KERNEL:
            p = self.world.params[k]
            c0, c1 = p.get('c0', 0), p.get('c1', 0)
            self.potential_FFT[k] = self.kernel_FFT[k] * self.world_FFT[c0]
            self.potential[k] = self.fftshift(np.real(self.ifftn(self.potential_FFT[k])))
            self.field[k] = gfunc(self.potential[k], p['m'], p['s'])
            if self.is_arita_mode or c1 in self.arita_layers:
                self.field[k] = (self.field[k] + 1) / 2
                D[c1] += dt * p['h'] * (self.field[k] - A[c1])
            else:
                D[c1] += dt * p['h'] * self.field[k]
            if not is_free_h: Dn[c1] += p['h']
        if not is_free_h:
            A_new = [A[c] + D[c] / Dn[c] if Dn[c]>0 else A[c] for c in CHANNEL]
        else:
            A_new = [A[c] + D[c] for c in CHANNEL]
        for c in CHANNEL:
            if self.add_noise > 0:
                rand = (np.random.random_sample(A_new[c].shape) - 0.5) * (self.add_noise/10) + 1
                A_new[c] *= rand
            if self.soft_clip_level > 0:
                A_new[c] = self.soft_clip(A_new[c], 0, 1)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
            else:
                A_new[c] = np.clip(A_new[c], 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
            if P > 0:
                A_new[c] = np.around(A_new[c] * P) / P
            self.change[c] = (A_new[c] - A[c]) / dt
            if is_update:
                if self.mask_rate > 0:
                    mask = np.random.random_sample(A_new[c].shape) > (self.mask_rate/10)
                    self.world.cells[c][mask] = A_new[c][mask]
                else:
                    self.world.cells[c] = A_new[c]
        if is_update:
            self.gen += 1
            self.time = round(self.time + dt, ROUND)
        # if self.is_gpu:
        #     self.gpu_thr.synchronize()

    def calc_kernel(self):
        R = self.world.model['R']
        dims = [slice(0, size) for size in SIZE]
        I = list(reversed(np.mgrid[list(reversed(dims))]))  # I, J, K, L
        self.X = [(i - mid) / R for i, mid in zip(I, MID)]  # X, Y, Z, S
        #self.D = np.sqrt(np.abs(sum([(-1)**d * x**2 for d,x in enumerate(self.X)])))  # Minkowski
        self.D = np.sqrt(sum([x**2 for x in self.X]))
        if DIM >= 3:
            Z = self.X[2]
            for d in range(3, DIM):
                Z = Z[MID[d]]
            self.Z_depth = Z - Z.min()
            self.Z_depth /= self.Z_depth.sum(axis=0) / 3

        if DIM == 2:
            ''' https://stackoverflow.com/questions/9924135/fast-cartesian-to-polar-to-cartesian-in-python '''
            #TH=[90, 360+90)=SIZE, R=[MIDX-1, -(MIDX-1)]=SIZE-1
            th_range = np.linspace(np.pi*1/2, np.pi*5/2, SIZETH+1)[:-1]
            r_range = np.arange(-SIZER+1, SIZER)[::-1]
            self.TH, self.R = np.meshgrid(th_range, r_range)
            self.polar_X = (self.R * np.cos(self.TH) + MIDX).astype(int)
            self.polar_Y = (self.R * np.sin(self.TH) + MIDY).astype(int)

        self.kernel = [self.kernel_shell(self.D, self.world.model, self.world.params[k]) for k in KERNEL]
        self.kernel_sum = [self.kernel[k].sum() for k in KERNEL]
        kernel_norm = [self.kernel[k] / self.kernel_sum[k] for k in KERNEL]
        self.kernel_FFT = [self.fftn(kernel_norm[k]) for k in KERNEL]
        self.kernel_updated = False

    def reset(self):
        self.gen = 0
        self.time = 0

class Analyzer:
    STAT_NAMES = {'p_m':'Param m', 'p_s':'Param s', 'n':'Gen (#)', 't':'Time (s)', 
        'm':'Mass (mg)', 'g':'Growth (mg/s)', 'r':'Gyradius (mm)',   # 'I':'Moment of inertia'
        'd':'Mass-growth distance (mm)', 's':'Speed (mm/s)', 'w':'Angular speed (deg/s)', 'm_a':'Mass asymmetry (mg)',
        'x':'X position(mm)', 'y':'Y position(mm)', 'l':'Lyapunov exponent',
        'k':'Rotational symmetry', 'w_k':'Rotational speed'}
        # 'k2':'Strength k=2', 'k3':'Strength k=3', 'k4':'Strength k=4', 'k5':'Strength k=5', 'k6':'Strength k=6',
        # 'w2':'Rotational k=2', 'w3':'Rotational k=3', 'w4':'Rotational k=4', 'w5':'Rotational k=5', 'w6':'Rotational k=6'}
        # 'm_r':'Mass on right (mg)', 'm_l':'Mass on left (mg)', 'a':'Semi-major axis (mm)', 'b':'Semi-minor axis (mm)', 'e':'Eccentricity', 'c':'Compactness', 'w_th':'Shape angular speed (deg/s)'}
    STAT_HEADERS = list(STAT_NAMES.keys())
    RECURRENCE_RANGE = slice(4, 11)
    SEGMENT_INIT = 128
    SEGMENT_INIT_LEN = 64
    SEGMENT_LEN_SHORT = 512
    SEGMENT_LEN_LONG = 2048
    PSD_INTERVAL = 32
    def get_stat_row(self):
        R, T = [self.world.model[k] for k in ('R', 'T')]
        pm, ps = [self.world.params[0][k] for k in ('m', 's')]
        if self.m_center is not None:
            pos = self.m_center * R + self.total_shift_idx
        else:
            pos = [0]*DIM
        RN = np.power(R, DIM)
        return [pm, ps, self.automaton.gen, self.automaton.time, 
                self.mass/RN, self.growth/RN, np.sqrt(self.inertia/self.mass) if self.mass!=0 else 0,  # self.inertia/RN  # self.inertia*RN, 
                self.mg_dist, self.m_shift*T, self.m_rotate*T, self.mass_asym/RN,
                pos[0], -pos[1], self.lyapunov,
                self.symm_sides, self.symm_rotate*T]
                # self.density_sum[2]/R, self.density_sum[3]/R, self.density_sum[4]/R, self.density_sum[5]/R, self.density_sum[6]/R,
                # self.rotate_wavg[2]*T, self.rotate_wavg[3]*T, self.rotate_wavg[4]*T, self.rotate_wavg[5]*T, self.rotate_wavg[6]*T]
                # self.mass_right/RN, self.mass_left/RN
                # self.shape_major_axis, self.shape_minor_axis,
                # self.shape_eccentricity, self.shape_compactness, self.shape_rotate]

    def __init__(self, automaton):
        self.automaton = automaton
        self.world = self.automaton.world
        # self.aaa = self.world.cells
        self.trim_segment = 1
        self.is_calc_symmetry = False
        self.is_calc_psd = False
        self.object_threshold = 0.05
        self.object_distance = 0.2 if CN==1 else 0.6
        self.make_border_mask()
        self.reset()

    def make_border_mask(self):
        A = self.world.cells[0]
        self.border_mask = np.full(A.shape, False, dtype=bool)
        for d in range(DIM):
            slices = [0 if d==d2 else slice(None) for d2 in range(DIM)]
            self.border_mask[tuple(slices)] = True
            slices = [A.shape[d]-1 if d==d2 else slice(None) for d2 in range(DIM)]
            self.border_mask[tuple(slices)] = True

    def reset(self):
        self.reset_values()
        self.reset_last()
        self.reset_position()
        self.reset_polar()
        self.clear_series()
        self.all_peaks = np.array([])
        self.good_peaks = np.array([])
        self.peak_mask = np.zeros(self.world.cells[0].shape, dtype=bool)
        self.peak_labels = np.zeros(self.world.cells[0].shape)
        self.object_map = np.zeros(self.world.cells[0].shape)
        self.object_border = np.zeros(self.world.cells[0].shape, dtype=bool)
        self.object_num = -1
        self.object_list = []

    def reset_values(self):
        self.is_empty = False
        self.is_full = False
        self.mass = 0
        self.growth = 0
        self.inertia = 0
        self.m_center = None
        self.g_center = None
        self.mg_dist = 0
        self.m_shift = 0
        self.m_angle = 0
        self.m_rotate = 0
        self.mass_asym = 0
        self.mass_right = 0
        self.mass_left = 0
        self.lyapunov = 0
        # self.shape_major_axis = 0
        # self.shape_minor_axis = 0
        # self.shape_eccentricity = 0
        # self.shape_compactness = 0
        # self.shape_angle = 0
        # self.shape_rotate = 0 

    def reset_last(self):
        self.m_last_center = None
        self.m_center = None
        self.m_last_angle = None
        # self.shape_last_angle = None

    def reset_position(self):
        self.last_shift_idx = np.zeros(DIM)
        self.total_shift_idx = np.zeros(DIM)

    def reset_polar(self):
        self.polar_array = None
        self.polar_avg = None
        self.polar_R = None
        self.polar_TH = None
        self.polar_FFT = None
        self.polar_density = None
        self.polar_angle = None
        self.polar_rotate = None
        self.last_polar_angle = None
        self.sides_vec = None
        self.angle_vec = None
        self.rotate_vec = None
        # self.density_vec = None
        self.density_sum = np.zeros((SIZEF))
        self.density_ema = None
        self.ema_alpha = 0.05
        # self.angle_wsum = None
        # self.angle_wavg = None
        self.rotate_wsum = None
        self.rotate_wavg = np.zeros((SIZEF))
        self.symm_sides = 0
        self.symm_angle = 0
        self.symm_rotate = 0

    def mode(self, arr):
        # l = list(arr)
        # s = set(l) - {0}
        # return max(s, key=l.count) if s else 0
        return max(arr, key=lambda x: (arr==x).sum() * (x>0))

    def calc_psd(self, X, fs, nfft=512, is_welch=True):
        if X is None or X == []:
            return None, None
        psd_func = scipy.signal.welch if is_welch else scipy.signal.periodogram
        freq, psd = psd_func(X, fs=fs, nfft=nfft, axis=0)
        half = len(freq)//2
        freq = freq[1:half]
        psd = psd[1:half]
        return freq, psd

    def robust_estimate(self, arr, mask):
        mask_n = np.sum(mask)
        if mask_n > 0:
            masked = arr[mask]
            return masked[0]  #outmost
            # return masked[mask_n // 2]  #median
            #Hodges-Lehmann estimator?
        else:
            return 0

    def calc_polar_FFT(self, polar_array, is_gaussian_blur=True):
        if is_gaussian_blur and PIXEL > 1:
            polar_array[:SIZER, :] = scipy.ndimage.filters.gaussian_filter(polar_array[:SIZER, :], sigma=(2,1))
        polar_FFT = self.automaton.fft1(polar_array[:SIZER, :])
        polar_FFT = polar_FFT[:SIZER, :SIZEF]
        polar_FFT[:, 0] = 0
        return polar_FFT

    def calc_stats(self, polar_what=0, psd_x='m', psd_y='g', is_welch=True):
        self.m_last_center = self.m_center
        self.m_last_angle = self.m_angle
        # self.shape_last_angle = self.shape_angle
        self.reset_values()

        R, T = [self.world.model[k] for k in ('R', 'T')]
        A = np.add.reduce(self.world.cells)
        G = np.maximum(np.add.reduce(self.automaton.field), 0)
        #focused auto-center
        # dr = int(R*1.7)
        # for d in range(DIM):
        #     slices = [slice(None,MID[d]-dr) if d==d2 else slice(None) for d2 in range(DIM)]
        #     A[tuple(slices)] = 0
        #     slices = [slice(MID[d]+dr,None) if d==d2 else slice(None) for d2 in range(DIM)]
        #     A[tuple(slices)] = 0

        X = self.automaton.X
        m0 = self.mass = A.sum()
        g0 = self.growth = G.sum()

        self.channel_alive = [(A0 > ALIVE_THRESHOLD).sum() for A0 in self.world.cells]
        self.border_alive = [(A0[self.border_mask] > ALIVE_THRESHOLD).sum() for A0 in self.world.cells]
        self.is_empty = any(a == 0 for a in self.channel_alive)
        self.is_full = sum(a > 0 for a in self.border_alive) > 0  #CN//3

        if m0 > EPSILON:
            AX = [A*x for x in X]
            MX1 = [ax.sum() for ax in AX]
            MX2 = [(ax*x).sum() for ax, x in zip(AX, X)]
            MX = self.m_center = np.asarray(MX1) / m0
            MuX2 = [mx2 - mx * mx1 for mx, mx1, mx2 in zip(MX, MX1, MX2)]
            self.inertia = sum(MuX2)

            # m11 = (AY*X).sum()
            # mu11 = m11 - mx * m10
            # m1 = mu20 + mu02
            # m2 = mu20 - mu02
            # m3 = 2 * mu11
            # t1 = m1 / 2 / m00
            # t2 = np.sqrt(m2**2 + m3**2) / 2 / m00
            # self.shape_major_axis = t1 + t2
            # self.shape_minor_axis = t1 - t2
            # self.shape_eccentricity = np.sqrt(1 - self.shape_minor_axis / self.shape_major_axis)
            # self.shape_compactness = m00 / (mu20 + mu02)
            # self.shape_angle = np.degrees(np.arctan2(m2, m3))
            # if self.shape_last_angle is not None:
                # self.shape_rotate = self.shape_angle - self.shape_last_angle
                # self.shape_rotate = (self.shape_rotate + 540) % 360 - 180

            if g0 > EPSILON:
                GX1 = [(G*x).sum() for x in X]
                GX = self.g_center = np.asarray(GX1) / g0
                self.mg_dist = np.linalg.norm(self.m_center - self.g_center)

            if self.m_last_center is not None and self.m_last_angle is not None:
                u = self.m_center
                v = self.m_last_center - self.last_shift_idx / R
                dm = u - v
                self.m_shift = np.linalg.norm(dm)
                self.m_angle = np.degrees(np.arctan2(dm[1], dm[0])) if self.m_shift >= EPSILON else 0
                # c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
                # self.m_angle = np.degrees(np.arccos(np.clip(c, -1, 1))) if self.m_shift >= EPSILON else 0
                self.m_rotate = self.m_angle - self.m_last_angle
                self.m_rotate = (self.m_rotate + 540) % 360 - 180
                if self.automaton.gen <= 2:
                    self.m_rotate = 0

                if DIM == 2:
                    midpoint = np.asarray([MIDX, MIDY])
                    X, Y = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
                    x0, y0 = self.m_last_center * R + midpoint - self.last_shift_idx
                    x1, y1 = self.m_center * R + midpoint
                    sign = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0)
                    self.mass_right = (A[sign>0]).sum()
                    self.mass_left = (A[sign<0]).sum()
                    self.mass_asym = self.mass_right - self.mass_left
                    # sign = (X - x0)
                    # self.mass_mir = (A[sign>0]).sum() - (A[sign<0]).sum()
                    # self.aaa = A.copy(); self.aaa[sign<0] = 0
            
            # self.lyapunov += ( np.log(abs(self.automaton.change.sum())) - self.lyapunov ) / self.automaton.gen

            if polar_what==0: A2 = self.world.cells
            elif polar_what==1: A2 = self.automaton.potential
            elif polar_what==2: A2 = self.automaton.field
            else: A2 = self.world.cells
            A2 = sum(A2)

            self.polar_array = A2[self.automaton.polar_Y, self.automaton.polar_X]
            if self.is_calc_symmetry:
                self.polar_avg = np.average(self.polar_array[:SIZER, :SIZEF], axis=1)
                self.polar_R = np.average(self.polar_array[:SIZER, :], axis=1)
                self.polar_TH = np.average(self.polar_array[:SIZER, :], axis=0)

                sides_row = np.arange(SIZEF).transpose()
                sides_row[0] = 1
                self.polar_FFT = self.calc_polar_FFT(self.polar_array, is_gaussian_blur=True)
                self.polar_density = np.abs(self.polar_FFT)
                self.polar_angle = np.angle(self.polar_FFT) / sides_row

                self.density_sum = np.sum(self.polar_density, axis=0)
                if self.density_ema is not None:
                    self.density_ema = self.density_ema + self.ema_alpha * (self.density_sum - self.density_ema)
                else:
                    self.density_ema = self.density_sum

                #adjust rotational angle and speed
                if self.last_polar_angle is not None:
                    if self.last_shift_idx[0] == self.last_shift_idx[1] == 0:
                        self.polar_rotate = self.polar_angle - self.last_polar_angle
                    else:
                        polar_array_unshift = A2[(self.automaton.polar_Y - self.last_shift_idx[1]) % SIZEY, (self.automaton.polar_X - self.last_shift_idx[0]) % SIZEX]
                        polar_FFT_unshift = self.calc_polar_FFT(polar_array_unshift, is_gaussian_blur=True)
                        polar_angle_unshift = np.angle(polar_FFT_unshift) / sides_row
                        self.polar_rotate = polar_angle_unshift - self.last_polar_angle

                    max_angle = np.pi/sides_row
                    # max_angle = np.pi
                    self.polar_rotate = (self.polar_rotate + 3*max_angle) % (2*max_angle) - max_angle

                    self.polar_rotate2 = self.polar_angle - self.last_polar_angle
                    self.polar_rotate2 = (self.polar_rotate2 + 3*max_angle) % (2*max_angle) - max_angle
                    self.polar_angle = self.last_polar_angle + self.polar_rotate2
                else:
                    self.polar_rotate = np.zeros(self.polar_FFT.shape)
                self.last_polar_angle = self.polar_angle

                #weighted by FFT density
                # self.angle_wsum = self.polar_angle * self.polar_density
                # self.angle_wavg = np.sum(self.angle_wsum, axis=0) / self.density_sum
                self.rotate_wsum = self.polar_rotate * self.polar_density
                self.rotate_wavg = np.sum(self.rotate_wsum, axis=0) / self.density_sum

                #per radius: symmetry, rotational angle and speed
                self.sides_vec = np.argmax(self.polar_density[:,2:SIZEF], axis=1)+2
                sides_idx = (np.arange(SIZER), self.sides_vec)
                # self.density_vec = self.polar_density[sides_idx]
                self.angle_vec = self.polar_angle[sides_idx] #/ self.sides_vec
                self.rotate_vec = self.polar_rotate[sides_idx]
                self.sides_vec[self.polar_avg < 0.05] = 0
                self.sides_vec[self.polar_avg > 0.95] = 0
                # self.sides_vec[self.sides_vec == 1] = 0
                # self.sides_vec[self.density_vec < 5] = 0

                #overall: symmetry
                #mode
                # self.symm_sides = self.mode(self.sides_vec)
                #max strength
                self.symm_sides = np.argmax(self.density_ema[2:SIZEF])+2

                #overall: rotational angle and speed
                #robust estimate
                mask = (self.sides_vec==self.symm_sides)
                self.symm_angle = self.robust_estimate(self.angle_vec, mask)
                self.symm_rotate = self.robust_estimate(self.rotate_vec, mask)
                #max strength
                # sides_ring = np.argmax(self.polar_density[:,self.symm_sides])
                # self.symm_angle = self.polar_angle[sides_ring, self.symm_sides] / self.symm_sides
                # self.symm_rotate = self.polar_rotate[sides_ring, self.symm_sides]
                #strength weighted
                # self.symm_angle = self.angle_wavg[self.symm_sides] / self.symm_sides
                # self.symm_rotate = self.rotate_wavg[self.symm_sides]

                #calc_period
            else:
                self.density_sum = np.zeros((SIZEF))
                self.rotate_wavg = np.zeros((SIZEF))

            if self.is_calc_psd:
                if self.series != []:
                    segment = self.series[-1]
                if self.series != [] and segment != []:
                    if self.automaton.gen % self.PSD_INTERVAL == 0:
                        X = np.asarray([val[psd_x] for val in segment])
                        # if psd_y == 'x': Y = np.asarray(self.series_R)
                        # if psd_y == 'y': Y = np.asarray(self.series_TH)
                        # else:
                        Y = np.asarray([val[psd_y] for val in segment])
                        self.psd_freq, self.psd1 = self.calc_psd(X, fs=T, nfft=512, is_welch=is_welch)
                        _, self.psd2 = self.calc_psd(Y, fs=T, nfft=512, is_welch=is_welch)
                        #if self.psd2 is not None: print(X.shape, self.psd1.shape, Y.shape, self.psd2.shape)

    def stats_fullname(self, i=None, x=None):
        if not x: x = self.STAT_HEADERS[i]
        return "{code}={name}".format(code=x, name=self.STAT_NAMES[x])

    def new_segment(self):
        if self.series == [] or self.series[-1] != []:
            self.series.append([])
    def clear_segment(self):
        if self.series != []:
            if self.series[-1] == []:
                self.series.pop()
            if self.series != []:
                self.series[-1] = []
        self.series_R = []
        self.series_TH = []
    def invalidate_segment(self):
        if self.series != []:
            self.series[-1] = [[self.world.params[0]['m'], self.world.params[0]['s']] + [np.nan] * (len(self.STAT_HEADERS)-2)]
            self.new_segment()
    def clear_series(self):
        self.current = None
        self.series = []
        self.series_R = []
        self.series_TH = []
        self.psd_freq = None
        self.psd1 = None
        self.psd2 = None
        self.period = None
        self.period_gen = 100

    def add_stats(self, psd_y='g'):
        multi = max(1, self.world.model['T'] // 10)
        if self.series == []:
            self.new_segment()
        segment = self.series[-1]
        self.current = self.get_stat_row()
        segment.append(self.current)
        if self.polar_R is not None:
            self.series_R.append(self.polar_R)
            self.series_TH.append(self.polar_TH)
        if self.trim_segment>0:
            if self.automaton.gen <= self.SEGMENT_INIT * multi:
                limit = self.SEGMENT_INIT_LEN * multi
            elif self.trim_segment in [1]:
                limit = self.SEGMENT_LEN_SHORT * multi
            elif self.trim_segment in [2]:
                limit = self.SEGMENT_LEN_LONG * multi
            while len(segment) > limit:
                segment.pop(0)
            while len(self.series_R) > limit:
                self.series_R.pop(0)
            while len(self.series_TH) > limit:
                self.series_TH.pop(0)

    def center_world(self):
        if self.mass < EPSILON or self.m_center is None:
            return
        axes = tuple(reversed(range(DIM)))
        self.last_shift_idx = (self.m_center * self.world.model['R']).astype(int)
        self.total_shift_idx += self.last_shift_idx
        self.world.cells = [np.roll(self.world.cells[c], -self.last_shift_idx, axes) for c in CHANNEL]
        self.automaton.potential = [np.roll(self.automaton.potential[k], -self.last_shift_idx, axes) for k in KERNEL]
        self.automaton.field = [np.roll(self.automaton.field[k], -self.last_shift_idx, axes) for k in KERNEL]
        self.automaton.change = [np.roll(self.automaton.change[c], -self.last_shift_idx, axes) for c in CHANNEL]
        # self.world.cells = scipy.ndimage.shift(self.world.cells, -self.last_shift_idx, order=0, mode='wrap')

    def detect_objects(self):
        '''
        peak_local_max: https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/peak.py
        ensure_spacing: https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/coord.py
        '''

        compact_watershed = 0.001
        blur = 0
        R = self.world.model['R']
        A = sum(self.automaton.potential) / len(self.automaton.potential)
        if KN == 1 and self.world.model.get('P') == 1:
            # cognitive domain of the glider in GoL
            for ii in range(2):
                fft = self.automaton.fftn(A)
                fft = self.automaton.kernel_FFT[0] * fft
                A = self.automaton.fftshift(np.real(self.automaton.ifftn(fft)))
        elif blur:
            A = scipy.ndimage.gaussian_filter(A, sigma=blur)
        A[A < 0.01] = 0

        # tile array by 3x3 to simulate periodic boundaries
        A_tiled = np.tile(A, tuple([3]*DIM))
        untile_slices = tuple([slice(size, size*2) for size in A.shape])

        # get all possible peaks (with periodic boundaries)
        min_distance = max(1, int(R*self.object_distance))
        footprint = np.ones((min_distance*2+1, ) * A.ndim, dtype=bool)
        self.all_peaks = skimage.feature.peak_local_max(A_tiled, min_distance=1, p_norm=2, footprint=footprint, exclude_border=1)
        # untile coordinates (keep those from center tile)
        keep = [ all(peak >= A.shape) and all(peak-A.shape < A.shape) for peak in self.all_peaks ]
        self.all_peaks = self.all_peaks[keep] - A.shape
        # choose peaks by ensuring minimum distance (same as peak_local_max(... min_distance=min_distance ...))
        self.good_peaks = skimage._shared.coord.ensure_spacing(self.all_peaks, spacing=min_distance, p_norm=2)

        # get peaks map
        self.peak_mask = np.zeros(A.shape, dtype=bool)
        self.peak_mask[tuple(self.good_peaks.T)] = True
        self.peak_labels, _ = scipy.ndimage.label(self.peak_mask)

        labels_tiled = np.tile(self.peak_labels, tuple([3]*DIM))

        # segmentation across borders
        self.object_map = skimage.segmentation.watershed(-A_tiled, labels_tiled, mask=A_tiled, compactness=compact_watershed)
        self.object_border = skimage.segmentation.find_boundaries(self.object_map, mode='inner')
        # untile arrays
        self.object_map = self.object_map[untile_slices]
        self.object_border = self.object_border[untile_slices]

        max_label = np.amax(self.object_map)
        self.object_list = []
        for label in range(1, max_label+1):
            self.object_list.append([self.world.cells[c][self.object_map == label] for c in CHANNEL])
        self.object_num = len(self.object_list)
        
        #RN = np.power(R, DIM)
        #print(self.object_num, " ".join(["{:.0f}".format(sum(ch.sum() for ch in obj)/RN) for obj in self.object_list]))

        # alternatives
        #self.peak_mask = np.logical_and(scipy.ndimage.maximum_filter(A, footprint=np.ones((3, 3))) == A, A>0)
        #self.peak_mask = skimage.feature.peak_local_max(A, indices=False, footprint=np.ones((3, 3)), min_distance=R)
        #self.peak_mask = skimage.morphology.local_maxima(A, indices=False, connectivity=R*R)
        #self.object_map = scipy.ndimage.watershed_ift(A.astype(np.uint8), self.peak_labels)
        #self.object_map = skimage.segmentation.random_walker(-A_tiled, labels_tiled, copy=False, beta=1, mode='cg_j')  #beta=10, mode='bf', default beta=130, mode='cg_j'

class Recorder:
    RECORD_ROOT = 'record'
    FRAME_EXT = '.png'
    VIDEO_EXT = '.mp4'
    GIF_EXT = '.gif'
    TIF_EXT = '.tif'
    GIF_FPS = 25
    VIDEO_FPS = 25
    ffmpeg_cmd = ['/usr/local/bin/ffmpeg',
        '-loglevel','warning', '-y',  # glocal options
        '-f','rawvideo', '-vcodec','rawvideo', '-pix_fmt','rgb24',  # input options
        '-s','{x}x{y}'.format(x=SIZEX*PIXEL, y=SIZEY*PIXEL), '-r',str(VIDEO_FPS),
        '-i','{input}',  # input pipe
        # '-an', '-vcodec','copy',  # output options (uncompressed 24-bit RGB, extremely large file, can no longer open in QuickTime)
        '-an', '-vcodec','libx264', '-pix_fmt','yuv420p', '-crf','1',  # output options (high quality H.264 encoding)
        '{output}']  # ouput file

    def __init__(self, world_list, is_save_gif):
        self.world_list = world_list
        self.is_save_gif = is_save_gif
        self.is_recording = False
        self.is_save_frames = False
        self.record_id = None
        self.record_seq = None
        self.img_dir = None
        self.video_path = None
        self.video = None
        self.gif_path = None
        self.gif = None

    def toggle_recording(self, is_save_frames=False):
        self.is_save_frames = is_save_frames
        if not self.is_recording:
            self.start_record()
            '''tif'''
            self.save_json(self.json_path)
            return True
        else:
            self.finish_record()
            return False

    def start_record(self):
        global STATUS
        ''' https://trac.ffmpeg.org/wiki/Encode/H.264
            https://trac.ffmpeg.org/wiki/Slideshow '''
        self.is_recording = True
        STATUS.append("> start " + ("saving frames" if self.is_save_frames else "recording video") + " and GIF...")
        self.record_id = '{name}-{time}'.format(name=self.world_list[0].names['code'].split('(')[0].replace('<','-').replace('?',''), time=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.record_seq = 1
        self.video_path = os.path.join(self.RECORD_ROOT, self.record_id + self.VIDEO_EXT)
        self.gif_path = os.path.join(self.RECORD_ROOT, self.record_id + self.GIF_EXT)
        self.tif_path = os.path.join(self.RECORD_ROOT, self.record_id + self.TIF_EXT)
        self.json_path = os.path.join(self.RECORD_ROOT, self.record_id + '.json')
        self.img_dir = os.path.join(self.RECORD_ROOT, self.record_id)
        if self.is_save_frames:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        else:
            cmd = [s.replace('{input}', '-').replace('{output}', self.video_path) for s in self.ffmpeg_cmd]
            try:
                '''tif'''
                self.video = subprocess.Popen(cmd, stdin=subprocess.PIPE)    # stderr=subprocess.PIPE
            except FileNotFoundError as e:
                self.video = None
                STATUS.append("> no ffmpeg program found!")
        self.gif = []
        '''tif'''
        # self.video = None

    def save_json(self, path):
        if len(self.world_list) == 1:
            A = copy.deepcopy(self.world_list[0])
            A.crop()
            data_list = [ A.to_data() ]
        else:
            A = [ copy.deepcopy(world) for world in self.world_list ]
            data_list = [ A0.to_data() for A0 in A ]
        try:
            with open(path, 'w', encoding='utf-8') as file:
                to_save = data_list if len(data_list) > 1 else data_list[0]
                json.dump(to_save, file, separators=(',', ':'), ensure_ascii=False)
                file.write('\n')
        except IOError as e:
            STATUS.append("I/O error({}): {}".format(e.errno, e.strerror))

    def save_image(self, img, filename=None):
        self.record_id = '{name}-{time}'.format(name=self.world_list[0].names['code'].split('(')[0], time=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        img_path = filename + self.FRAME_EXT if filename else os.path.join(self.RECORD_ROOT, self.record_id + self.FRAME_EXT)
        img.save(img_path)

    def record_frame(self, img):
        if self.is_save_frames:
            img_path = os.path.join(self.RECORD_ROOT, self.record_id, '{:03d}'.format(self.record_seq) + self.FRAME_EXT)
            img.save(img_path)
        else:
            if self.video:
                img_rgb = img.convert('RGB').tobytes()
                self.video.stdin.write(img_rgb)
        # img.convert('P', dither=0)
        self.gif.append(img)
        self.record_seq += 1

    def finish_record(self):
        global STATUS
        if self.is_save_frames:
            STATUS.append("> frames saved to '" + self.img_dir + "/*" + self.FRAME_EXT + "'")
            cmd = [s.replace('{input}', os.path.join(self.img_dir, '%03d'+self.FRAME_EXT)).replace('{output}', self.video_path) for s in self.ffmpeg_cmd]
            try:
                subprocess.call(cmd)
            except FileNotFoundError as e:
                self.video = None
                STATUS.append("> no ffmpeg program found!")
        else:
            if self.video:
                self.video.stdin.close()
                STATUS.append("> video saved to '" + self.video_path + "'")

        if self.is_save_gif:
            durations = [1000 // self.GIF_FPS] * len(self.gif)
            durations[-1] *= 10
            '''tif'''
            self.gif[0].save(self.gif_path, format=self.GIF_EXT.lstrip('.'), save_all=True, append_images=self.gif[1:], loop=0, duration=durations)
            # self.gif[0].save(self.tif_path, save_all=True, append_images=self.gif[1:])
            self.gif = None
            STATUS.append("> GIF saved to '" + self.gif_path + "'")

        self.is_recording = False

'''
class CPPN:
    def __init__(self, X, Y, D, z_size=8, scale=1, net_size=32, variance=None):
        self.X = X.reshape(-1, 1) * scale
        self.Y = Y.reshape(-1, 1) * scale
        self.D = D.reshape(-1, 1) * scale
        self.z_size = z_size
        self.scale = scale
        self.net_size = net_size
        self.variance = variance
        self.init_model()

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    @staticmethod
    def modulus(x): return np.mod(x, 5)

    def init_model(self):
        s = self.variance or np.random.uniform(0.5, 5)
        x = 3 + self.z_size
        n = self.net_size
        self.model_W = [];  self.model_B = [];  self.model_F = []
        self.model_W.append(np.random.normal(0, 3, (x, n)));  self.model_B.append(np.zeros((n,)));  self.model_F.append(np.tanh)
        self.model_W.append(np.random.normal(0, 1, (n, n)));  self.model_B.append(np.zeros((n,)));  self.model_F.append(np.tanh)
        self.model_W.append(np.random.normal(0, 1, (n, n)));  self.model_B.append(np.zeros((n,)));  self.model_F.append(np.tanh)
        self.model_W.append(np.random.normal(0, 0.1, (n, 1)));  self.model_B.append(np.zeros((1,)));  self.model_F.append(self.sigmoid)
        # for W in self.model_W: print(W.shape, np.amax(W), np.amin(W))

    def generate_z(self):
        return np.random.normal(0, 1, self.z_size)

    def generate(self, z=None):
        size = self.X.shape[0]
        Z = np.repeat(z, size).reshape((-1, size))
        input = np.concatenate([self.X, self.Y, self.D, Z.T], axis=1)
        A = input
        for W, B, F in zip(self.model_W, self.model_B, self.model_F):
            A = F(np.matmul(A, W) + B)
        return A.reshape((SIZEX, SIZEY))
'''

class Lenia:
    MARKER_COLORS_W = [0x5F,0x5F,0x5F,0x7F,0x7F,0x7F,0xFF,0xFF,0xFF]
    MARKER_COLORS_B = [0x9F,0x9F,0x9F,0x7F,0x7F,0x7F,0x0F,0x0F,0x0F]
    POLYGON_NAME = {1:'irregular', 2:'bilateral', 3:'trimeric', 4:'tetrameric', 5:'pentameric', 
        6:'hexameric', 7:'heptameric', 8:'octameric', 9:'nonameric', 10:'decameric', 0:'polymeric'}
    SAVE_ROOT = 'save'
    ANIMALS_PATH = 'animals.json' if DIM==2 else 'animals'+str(DIM)+'D.json'
    found_path = args.F
    if found_path is None:
        found_path = '{D}{C}{K}{X}.json'.format(D=DIM, C=CN, K=KN, X='' if XN==1 else str(XN))
    elif not found_path.endswith('.json'):
        found_path += '.json'
    FOUND_ANIMALS_PATH = 'found/' + found_path
    #SOFT_CLIP_NAME_LIST = ["Off","tanh","log 10000","log 1000","log 100","log 10"]
    SOFT_CLIP_NAME_LIST = ["Off","tanh","exp 11","exp 10","exp 9","exp 8","exp 7","exp 6","exp 5","exp 4"]

    def __init__(self):
        self.is_run = True
        self.run_counter = -1
        self.is_closing = False
        self.is_advanced_menu = False

        self.show_what = 0
        self.show_group = 0  # avg, channels, each kernel
        self.show_kernel = 0
        self.polar_mode = 0
        self.markers_mode = 1

        self.stats_mode = 0
        self.stats_x = 4
        self.stats_y = 5
        self.is_group_params = False
        self.is_draw_params = False
        self.is_auto_center = False
        self.auto_rotate_mode = 0

        self.is_show_fps = False
        self.fps = None
        self.last_time = None

        self.fore = None
        self.back = None
        self.is_layer_mode = False
        self.is_auto_load = False
        self.last_seed = self.random_hex()
        self.random_density = 40

        self.search_mode = None
        self.is_search_small = False
        self.search_algo = 4  # 0=global, 1=depth, 2=breadth, 3=depth+breadth, 4=GA(avg), 5=GA(stdev), 6=GA(max)
        self.breadth_count = 0
        self.is_show_search = False
        self.search_stage = 0
        self.search_total = 0
        self.search_success = 0
        self.search_back = None
        self.search_back2 = None
        self.leaderboard_size = 10
        self.leaderboard = [{'fitness':float('-inf'), 'world':None} for i in range(self.leaderboard_size)]
    
        self.is_show_slice = False
        self.z_slices = [MID[DIM-1-d] for d in range(DIM-2)]  # S, Z
        self.z_axis = DIM-3

        ''' http://hslpicker.com/ '''
        self.colormaps = [
            self.create_colormap_turbo(seq='rgb'),
            self.create_colormap_turbo(seq='grb'),
            self.create_colormap(np.asarray([[7,6,7],[5,4,5],[4,1,4],[1,3,6],[3,4,6],[4,5,7],[2,6,3],[5,6,4],[6,7,5],[8,8,3],[8,6,2],[8,5,1],[7,0,0]])), #Paul Tol's Rainbow
            self.create_colormap(np.asarray([[0,0,4],[0,0,8],[0,4,8],[0,8,8],[4,8,4],[8,8,0],[8,4,0],[8,0,0],[4,0,0]])), #BCYR = Jet
            self.create_colormap(np.asarray([[0,2,0],[0,4,0],[4,6,0],[8,8,0],[8,4,4],[8,0,8],[4,0,8],[0,0,8],[0,0,4]])), #GYPB
            self.create_colormap(np.asarray([[4,0,2],[8,0,4],[8,0,6],[8,0,8],[4,4,4],[0,8,0],[0,6,0],[0,4,0],[0,2,0]])), #PPGG
            # self.create_colormap(np.asarray([[4,4,6],[2,2,4],[2,4,2],[4,6,4],[6,6,4],[4,2,2]])), #BGYR
            # self.create_colormap(np.asarray([[4,6,4],[2,4,2],[4,4,2],[6,6,4],[6,4,6],[2,2,4]])), #GYPB
            # self.create_colormap(np.asarray([[6,6,4],[4,4,2],[4,2,4],[6,4,6],[4,6,6],[2,4,2]])), #YPCG
            self.create_colormap(np.asarray([[8,8,8],[7,7,7],[5,5,5],[3,3,3],[0,0,0]]), is_marker_w=False), #W/B
            self.create_colormap(np.asarray([[0,0,0],[3,3,3],[5,5,5],[7,7,7],[8,8,8]]))] #B/W
            # self.create_colormap_flat(255, 218, is_marker_w=False)]  #flat grey
        self.colormap_id = 0
        
        # self.color_list = {
        #     'R':[8,0,0], 'G':[0,8,0], 'B':[0,0,8],  # RGB = Red/Green/Blue
        #     'O':[5,4,0], 'T':[0,5,4], 'V':[4,0,5],  # OTV = Orange/Turquoise/Violet
        #     'W':[4,4,4], 'E':[3,3,3], 'K':[2,2,2],  # Greyscale
        #     '1':[1,1,1], '2':[2,2,2], '3':[3,3,3], '4':[4,4,4], '5':[5,5,5], '6':[6,6,6], '3':[7,7,7], }
        # self.channelmaps = ['RGBOTV', 'RBGOVT', 'OTVRGB', 'OVTRBG', '432765']
        self.channelmaps = np.asarray([
            [[8,0,0],[0,8,0],[0,0,8],[0,5,4],[5,4,0],[4,0,5]], # RGB = Red/Green/Blue
            #[[8,0,0],[0,8,0],[0,0,8],[4,0,5],[5,4,0],[0,5,4]], # RGB = Red/Green/Blue
            [[8,0,0],[0,0,8],[0,8,0],[5,4,0],[4,0,5],[0,5,4]], # RBG = Red/Blue/GREEN
            [[5,4,0],[0,5,4],[4,0,5],[8,0,0],[0,8,0],[0,0,8]], # OTV = Orange/Turquoise/Violet or [6,3,0]
            [[5,4,0],[4,0,5],[0,5,4],[8,0,0],[0,0,8],[0,8,0]], # OVT = Orange/Violet/Turquoise
            [[7,7,7],[6,6,6],[5,5,5],[4,4,4],[3,3,3],[2,2,2]]  # Greyscale
            # [[8,0,0],[0,0,0],[0,0,0]], # Red only
            # [[0,0,0],[0,8,0],[0,0,0]], # Green only
            # [[0,0,0],[0,0,0],[0,0,8]], # Blue only
            # [[4,4,4],[3,3,3],[2,2,2]]  # Greyscale
        ]) / 8
        self.channelbg = np.asarray([
            [0,0,2], [0,0,2], [0,1,0], [0,1,0], [0,0,0]
            #[0,0,0], [0,0,0], [0,0,0]
        ]) / 8
        self.channel_group = 0
        self.channel_shift = 0

        self.last_key = None
        self.excess_key = None
        self.info_type = 'animal'
        self.clear_job = None
        self.clipboard_st = ""
        self.is_save_image = False
        self.file_seq = 0

        self.samp_freq = 1
        self.samp_gen = 1
        self.samp_rotate = 0
        self.is_samp_clockwise = False
        self.samp_sides = 1

        self.animal_id = 0
        self.found_animal_id = 0
        self.last_load_animal = True
        self.deleted_found_animal_count = 0
        self.animal_data = []
        self.found_animal_data = []
        self.read_animals()
        self.read_found_animals()

        self.world_list = [ Board(list(reversed(SIZE))) ]
        self.world = self.world_list[0]
        self.blank_channel = np.zeros(self.world.cells[0].shape)
        # self.target = Board((SIZEY, SIZEX))
        self.automaton_list = [ Automaton(world) for world in self.world_list ]
        self.automaton = self.automaton_list[0]
        self.analyzer = Analyzer(self.automaton_list[0])
        self.recorder = Recorder(self.world_list, is_save_gif=True) #is_save_gif=not self.is_show_rgb())
        self.clear_transform()
        self.create_window()
        self.create_menu()
        #self.cppn = CPPN(self.automaton.X, self.automaton.Y, self.automaton.Z, self.automaton.S, self.automaton.D, z_size=8, scale=2, net_size=64)
        #self.font = PIL.ImageFont.truetype('resource/monaco.ttf', 10)
        #self.convert_font_run_once('resource/bitocra-13.bdf')
        self.font = PIL.ImageFont.load('resource/bitocra-13.pil')

    @property
    def stats_x_name(self): return self.analyzer.STAT_HEADERS[self.stats_x]
    @property
    def stats_y_name(self): return self.analyzer.STAT_HEADERS[self.stats_y]
    @stats_x_name.setter
    def stats_x_name(self, val): self.stats_x = self.analyzer.STAT_HEADERS.index(val)
    @stats_y_name.setter
    def stats_y_name(self, val): self.stats_y = self.analyzer.STAT_HEADERS.index(val)

    def convert_font_run_once(self, font_file_path):
        import PIL.BdfFontFile, PIL.PcfFontFile
        ''' https://stackoverflow.com/questions/48304078/python-pillow-and-font-conversion '''
        ''' https://github.com/ninjaaron/bitocra '''
        with open(font_file_path, 'rb') as fp:
            p = PIL.BdfFontFile.BdfFontFile(fp) #PcfFontFile if you're reading PCF files
            p.save(font_file_path)

    def clear_transform(self):
        self.tx = {'shift':[0]*DIM, 'rotate':[0]*3, 'R':self.world.model['R'], 'flip':-1}

    def read_animals(self):
        self.has_animal_data = False
        try:
            with open(self.ANIMALS_PATH, 'r', encoding='utf-8') as file:
                new_animal_data = json.load(file)
                new_animal_data = [line for line in new_animal_data if type(line) in [dict]]
                self.animal_data = new_animal_data
            self.has_animal_data = self.animal_data != []
        except IOError:
            pass
        except json.JSONDecodeError as e:
            STATUS.append("> JSON file error")
            print(e)

    def read_found_animals(self):
        try:
            with open(self.FOUND_ANIMALS_PATH, 'r', encoding='utf-8') as file:
                st = file.read()
                st = "[" + st.rstrip(', \n\r\t') + "]"
                new_found_animal_data = json.loads(st)
                new_found_animal_data = [line for line in new_found_animal_data if type(line) in [dict]]
                self.found_animal_data = new_found_animal_data
                self.found_animal_id -= self.deleted_found_animal_count
                self.deleted_found_animal_count = 0
                STATUS.append("> found lifeforms loaded from " + self.FOUND_ANIMALS_PATH)
        except IOError:
            pass
        except json.JSONDecodeError as e:
            print(e)

    def delete_found_animal(self, code):
        try:
            lines = open(self.FOUND_ANIMALS_PATH, 'r', encoding='utf-8').readlines()
            with open(self.FOUND_ANIMALS_PATH, 'w', encoding='utf-8') as file:
                for line in lines:
                    if not (line.startswith('{"code":"' + code + '",') and '"cells":' in line):
                        file.write(line)
            self.found_animal_id += 1
            self.deleted_found_animal_count += 1
        except IOError:
            pass
        except json.JSONDecodeError as e:
            print(e)
        # self.read_found_animals()

    def load_animal_id(self, world, id, **kwargs):
        if not self.has_animal_data:
            return
        self.animal_id = max(0, min(len(self.animal_data)-1, id))
        self.load_part(world, Board.from_data(self.animal_data[self.animal_id]), **kwargs)
        self.last_load_animal = True

    def load_found_animal_id(self, world, id, **kwargs):
        if self.found_animal_data is None or self.found_animal_data == []:
            return
        self.found_animal_id = max(0, min(len(self.found_animal_data)-1, id))
        self.load_part(world, Board.from_data(self.found_animal_data[self.found_animal_id]), is_use_part_R=True, **kwargs)
        if self.world.names['code'] == '':
            self.world.names['code'] = ['Found #' + str(self.found_animal_id + 1)]
        self.last_load_animal = False

    def load_animal_code(self, world, code, **kwargs):
        if not self.has_animal_data:
            return
        if not code: return
        id = self.get_animal_id(code)
        if id is not None and id != -1:
            self.load_animal_id(world, id, **kwargs)
        return id

    def load_found_animal_code(self, world, code, **kwargs):
        if self.found_animal_data is None or self.found_animal_data == []:
            return
        if not code: return
        id = self.get_found_animal_id(code)
        if id is not None and id != -1:
            self.load_found_animal_id(world, id, **kwargs)
        return id

    def get_animal_id(self, code):
        if not self.has_animal_data:
            return -1
        code_sp = code.split(':')
        n = int(code_sp[1]) if len(code_sp)==2 else 1
        itr = (id for (id, data) in enumerate(self.animal_data) if data["code"]==code_sp[0])
        for i in range(n):
            id = next(itr, None)
        return id

    def get_found_animal_id(self, code):
        if self.found_animal_data is None or self.found_animal_data == []:
            return -1
        code_sp = code.split('<')
        itr = (id for (id, data) in enumerate(self.found_animal_data) if data["code"]==code_sp[0] or data["code"].startswith(code_sp[0]+'<'))
        id = next(itr, None)
        return id

    def search_animal_id(self, prefix, old_id, dir):
        if not self.has_animal_data:
            return -1
        id = old_id + dir
        while id >= 0 and id < len(self.animal_data):
            if self.animal_data[id]["name"].startswith(prefix):
                return id
            else:
                id += dir
        return old_id

    def search_found_animal_id(self, prefix, old_id, dir):
        if self.found_animal_data is None or self.found_animal_data == []:
            return -1
        id = old_id + dir
        while id >= 0 and id < len(self.found_animal_data):
            if self.found_animal_data[id]["name"].startswith(prefix):
                return id
            else:
                id += dir
        return old_id

    def search_animal(self, world, prefix, dir):
        if not self.has_animal_data:
            return
        id = self.animal_id
        if dir == +1:
            id = self.search_animal_id(prefix, id, dir)
        elif dir == -1:
            id = self.search_animal_id(prefix, id, dir)
            id = self.search_animal_id(prefix, id, dir)
        while id < len(self.animal_data) and self.animal_data[id]["code"].startswith(">"):
            id += 1
        self.load_animal_id(world, id)

    def search_found_animal(self, world, prefix, dir):
        if self.found_animal_data is None or self.found_animal_data == []:
            return
        id = self.search_found_animal_id(prefix, self.found_animal_id, dir)
        self.load_found_animal_id(world, id)

    def load_part(self, world, part, is_replace=True, is_use_part_R=False, is_random=False, is_auto_load=False, repeat=1):
        if part is None:
            return
        self.fore = part
        if self.is_layer_mode:
            # if world.names['code'] != part.names['code']:
            print(world.names['code'], part.names['code'])
        if part.names is not None and part.names['code'].startswith('~'):
            part.names['code'] = part.names['code'].lstrip('~')
            world.model['R'] = part.model['R']
            self.automaton.calc_kernel()
        if part.names is not None and is_replace:
            world.names = part.names.copy()
        if part.cells is not None:
            if part.params is None:
                part.params = world.params
            is_life = (world.model.get('P') == 1)
            will_be_life = (part.model.get('P') == 1)
            if not is_life and will_be_life:
                self.colormap_id = len(self.colormaps) - 1
                self.window.title('Conway\'s Game of Life')
            elif is_life and not will_be_life:
                self.colormap_id = 0
                world.model['R'] = DEF_R
                self.automaton.calc_kernel()
                self.window.title("Lenia {d}D".format(d=DIM))
            if self.is_layer_mode:
                self.back = copy.deepcopy(world)
            if is_replace and not self.is_layer_mode:
                if not is_auto_load:
                    if is_use_part_R:
                        R = part.model['R']
                    else:
                        R = world.model['R']
                    world.model = copy.deepcopy(part.model)
                    world.model['R'] = R
                    world.params = copy.deepcopy(part.params)
                    world.settings = copy.deepcopy(part.settings)
                    self.automaton.calc_kernel()
                if 'clip' in world.settings and world.settings['clip'] in self.SOFT_CLIP_NAME_LIST:
                    self.automaton.soft_clip_level = self.SOFT_CLIP_NAME_LIST.index(world.settings['clip'])
                if 'arita' in world.settings:
                    self.automaton.arita_layers = world.settings['arita']
                else:
                    self.automaton.arita_layers = []
                world.clear()
                self.automaton.reset()
                if is_auto_load:
                    self.analyzer.reset_position()
                    self.analyzer.reset_values()
                else:
                    self.analyzer.reset()
            self.clear_transform()
            for i in range(repeat):
                if is_random:
                    is_life = (world.model.get('P') == 1)
                    if is_life:
                        self.tx['rotate'] = [np.random.randint(4) * 90, 0, 0]
                    else:
                        self.tx['rotate'] = (np.random.rand(3) * 360).tolist()
                    shape1 = world.cells[0].shape
                    shape0 = min(part.cells[0].shape, world.cells[0].shape)
                    self.tx['shift'] = [np.random.randint(d1 + d0) - d1//2 for d0, d1 in zip(shape0, shape1)]
                    self.tx['flip'] = np.random.randint(3) - 1
                world.add_transformed(part, self.tx)

    def check_auto_load(self):
        if self.is_auto_load:
            self.load_part(self.world, self.fore, is_auto_load=True)
        else:
            self.automaton.reset()

    def transform_world(self):
        if self.is_layer_mode:
            if self.back is not None:
                self.world.cells = copy.deepcopy(self.back.cells)
                self.world.model = copy.deepcopy(self.back.model)
                self.world.params = copy.deepcopy(self.back.params)
                self.world.transform(self.tx, mode='Z', z_axis=self.z_axis, is_world=True)
                self.world.add_transformed(self.fore, self.tx)
                self.analyzer.reset()
        else:
            if not self.is_run:
                if self.back is None:
                    self.back = copy.deepcopy(self.world)
                else:
                    self.world.cells = copy.deepcopy(self.back.cells)
                    self.world.model = copy.deepcopy(self.back.model)
                    self.world.params = copy.deepcopy(self.back.params)
            if self.tx['flip'] < 2:
                self.world.transform(self.tx, z_axis=self.z_axis, is_world=True)
            else:
                self.world.transform(self.tx, mode='RS', z_axis=self.z_axis, is_world=True)
                self.world.transform(self.tx, mode='ZF', z_axis=self.z_axis, is_world=True)
            self.automaton.calc_kernel()
            self.analyzer.reset_last()

    def world_updated(self, is_random=False):
        if not self.is_layer_mode:
            self.back = copy.deepcopy(self.world)
        self.automaton.reset()
        if not is_random:
            self.analyzer.reset()

    def clear_world(self):
        self.world.clear()
        self.world_updated()
        self.world.names = {'code':'', 'name':'empty', 'cname':''}

    code_list = list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')  # no I,O, ambiguous with 1,0
    hex_list = list('0123456789ABCDEF')
    def random_code(self, size=6):
        return ''.join(np.random.choice(self.code_list, size=size))
    def random_hex(self, size=8):
        return ''.join(np.random.choice(self.hex_list, size=size))

    def random_world(self, is_reseed=False, is_fill=False, density_mode=1):
        R = self.world.model['R']
        # if is_reseed and self.last_seed is not None:
        #     np.random.set_state(self.last_seed)
        # else:
        #     self.last_seed = np.random.get_state()
        if is_reseed:
            if self.world.names['cname'].startswith('seed:'):
                self.last_seed = self.world.names['cname'].split(':')[-1]
        else:
            self.last_seed = self.random_hex()
        try:
            seed = int(self.last_seed, 16)
        except:
            print('error in seed, use 0')
            seed = 0
        np.random.seed(seed)

        if is_fill:
            dims = [size - R*2 for size in SIZE]
            rand = [np.random.random_sample(tuple(reversed(dims))) * 0.9 for c in CHANNEL]
            self.world.clear()
            self.world.add(Board.from_values(rand))
        else:
            self.world.clear()
            dim = int(R * 0.9)
            dims = [dim]*DIM
            self.random_density = 0.75 * np.prod(SIZE) / R**DIM
            density = min(max(int(self.random_density), 10), 1000)
            if density_mode==0: density //= 2
            elif density_mode==2: density *= 2
            for i in range(density):
                rand = [np.random.random_sample(tuple(reversed(dims))) * 0.9 for c in CHANNEL]
                border = int(R * 0.5)
                shift = [ [np.random.randint(border, size - border) - size//2 if border < size - border else 0  for size in reversed(SIZE)] for c in CHANNEL]
                self.world.add(Board.from_values(rand), shift=shift)

        self.world_updated()
        self.world.names['name'] = ''
        self.world.names['cname'] = 'seed:'+str(self.last_seed)

    def random_local(self, p, i, delta, vmin, vmax, digit):
        p[i] += np.random.rand() * (delta*2) - delta
        p[i] = max(vmin, min(vmax, round(p[i], digit)))
    def random_global(self, p, i, vmin, vmax, digit):
        p[i] = np.random.rand() * (vmax-vmin) + vmin
        p[i] = round(p[i], digit)
    def random_global_mul(self, p, i, i0, dmin, dmax, digit):
        d = np.random.rand() * (dmax-dmin) + dmin
        p[i] = p[i0] / 10 * d
        p[i] = round(p[i], digit)
 
    def random_params(self, is_incremental=False):
        # R = np.random.randint(15, 35)
        if is_incremental:  # local
            is_small = np.random.randint(5) == 0
            #change_channel = np.random.randint(0, CN)
            B_change =  0 if is_small else 25
            b_change = 25 if is_small else 5
            # A_change = 5
            # rand = np.random.randint(A_change) if A_change>0 else -1
            # if rand in [0]:
            #     if 'arita' not in self.world.settings:
            #         self.world.settings['arita'] = []
            #     arita = self.world.settings['arita']
            #     ch = np.random.randint(CN)
            #     arita.append(ch) if ch not in arita else arita
            #     self.automaton.arita_layers = arita
            for k in KERNEL:
                p = self.world.params[k]
                c0, c1 = p.get('c0', 0), p.get('c1', 0)
                scale = 1
                #if is_small or c1 == change_channel:
                rand = np.random.randint(b_change) if b_change>0 else -1
                if rand in [0,1] and len(p['rings']) < 3:
                    p['rings'].append(EMPTY_RING.copy())
                elif rand in [2] and len(p['rings']) > 1:
                    p['rings'].pop()
                for ring in p['rings']:
                    rand = np.random.randint(b_change) if b_change>0 else -1
                    if rand == 0:
                        self.random_local(ring, 'r', delta=scale*0.01 if is_small else scale*0.1, vmin=0.01, vmax=1.0, digit=2)
                        self.random_local(ring, 'w', delta=scale*0.01 if is_small else scale*0.1, vmin=0.01, vmax=1.0, digit=2)
                        self.random_local(ring, 'b', delta=scale*0.01 if is_small else scale*0.1, vmin=0.0, vmax=1.0, digit=2)
                self.random_local(p, 'm', delta=scale*0.02  if is_small else scale*0.05,  vmin=0.1,  vmax=1.0, digit=3)
                self.random_local(p, 's', delta=scale*0.002 if is_small else scale*0.005, vmin=0.01, vmax=1.0, digit=4)
                self.random_local(p, 'h', delta=scale*0.02  if is_small else scale*0.1,   vmin=0.1,  vmax=1.0, digit=2)
        else:  # global
            for k in KERNEL:
                p = self.world.params[k]
                new_R = np.random.randint(RAND_R1, RAND_R2) if RAND_R1 < RAND_R2 else RAND_R1
                self.world.model['R'] = new_R
                p['rings'] = []
                for b in range(np.random.randint(3, 4)):
                    ring = {}
                    self.random_global(ring, 'r', vmin=0.01, vmax=1.0, digit=2)
                    self.random_global(ring, 'w', vmin=0.01, vmax=1.0, digit=2)
                    self.random_global(ring, 'b', vmin=0.0, vmax=1.0, digit=2)
                    p['rings'].append(ring)
                self.random_global(p, 'm', vmin=0.1, vmax=0.5, digit=3)
                self.random_global_mul(p, 's', 'm', dmin=1/2, dmax=3, digit=4)  #1/1.5, 2.5 | 1, 5
                self.random_global(p, 'h', vmin=0.1, vmax=1.0, digit=2)
                # p['r'] = 1
        self.automaton.calc_kernel()
        self.world_updated()
        self.update_lineage()

    '''
    def cppn_world(self, is_reseed=False):
        if is_reseed and self.last_seed is not None:
            np.random.set_state(self.last_seed)
        else:
            self.last_seed = np.random.get_state()

        CPPNX, CPPNY, CPPNZ, CPPNS = SIZEX*3//4, SIZEY*3//4, SIZEZ*3//4, SIZES*3//4
        self.z = self.cppn.generate_z()
        part = self.cppn.generate(self.z)
        self.world.clear()
        self.world.add(Board.from_values(part[0:CPPNS, 0:CPPNZ, 0:CPPNY, 0:CPPNX]))
        self.world_updated()
    '''

    def backup_world(self, i=1, is_reset=True):
        if i==1:
            self.search_back = copy.deepcopy(self.world)
        elif i==2:
            self.search_back2 = copy.deepcopy(self.world)
        if is_reset:
            self.automaton.reset()
            self.analyzer.reset()

    def restore_world(self, i=1):
        if i==1:
            back = self.search_back
        elif i==2:
            back = self.search_back2
        self.world.cells = copy.deepcopy(back.cells)
        self.world.model = copy.deepcopy(back.model)
        self.world.params = copy.deepcopy(back.params)
        self.world.names = copy.deepcopy(back.names)
        self.automaton.reset()
        self.analyzer.reset()

    def put_world_in_leaderboard(self):
        multi = max(1, self.world.model['T'] // 10)
        start_gen = self.analyzer.SEGMENT_INIT * multi
        seg = self.analyzer.series[-1][start_gen:]
        val_seg = [val[self.stats_x] for val in seg]
        if self.search_algo in [4]:
            fitness = sum(val_seg) / len(val_seg)
        elif self.search_algo in [5]:
            avg = sum(val_seg) / len(val_seg) 
            var = sum([((x - avg) ** 2) for x in val_seg]) / len(val_seg)
            fitness = var ** 0.5
        elif self.search_algo in [6]:
            fitness = max(val_seg)
        func = ["avg","stdev","max"][self.search_algo-4]
        cname = "{func}({stat})={fitness:.3f}".format(func=func, stat=self.stats_x_name, fitness=fitness)
        self.world.names['cname'] = cname
        # self.update_lineage(cname=cname)
        self.leaderboard.append({'fitness':fitness, 'world':copy.deepcopy(self.world)})
        self.leaderboard.sort(key=lambda entity: entity['fitness'], reverse=True)
        self.leaderboard.pop()

    def mutate_world_from_leaderboard(self):
        # print(["{:.3f} {}".format(e['val'], e['world'].names['code'] if e['world'] is not None else 'X') for e in self.leaderboard])
        order = np.random.permutation(self.leaderboard_size)
        # if np.random.randint(5) == 0:
        #     #recombine
        #     w1 = None
        #     w2 = None
        #     for i in order:
        #         world = self.leaderboard[i]['world']
        #         if world is not None:
        #             if w1 is None: w1 = world
        #             elif w2 is None: w2 = world
        #             else:
        #                 self.world.params = []
        #                 self.world.names = [self.clean_code(w1.names['code'])+"x"+self.clean_code(w2.names['code']), '', '']
        #                 self.update_lineage()
        #                 for c in CHANNEL:
        #                     w = w1 if np.random.randint(2) == 0 else w2
        #                     self.world.cells[c] = copy.deepcopy(w.cells[c])
        #                     for k in KERNEL:
        #                         p = w.params[k]
        #                         if p['c0'] == c:
        #                             self.world.params.append(p)
        #                 return
        #mutate
        for i in order:
            world = self.leaderboard[i]['world']
            if world is not None:
                self.world.cells = copy.deepcopy(world.cells)
                self.world.settings = copy.deepcopy(world.settings)
                self.world.model = copy.deepcopy(world.model)
                self.world.params = copy.deepcopy(world.params)
                self.world.names = copy.deepcopy(world.names)
                self.random_params(is_incremental=True)
                return
        STATUS.append("search ended prematurely")
        self.finish_search()

    def toggle_search(self, search_mode):
        if self.search_mode is None:
            self.search_mode = search_mode
            self.start_search()
        else:
            self.finish_search()

    def start_search(self):
        if self.search_mode == 0:
            self.samp_freq = 1
            self.search_stage = 0
            self.automaton.gen = 0
            self.search_total = 0
            self.search_success = 0
            # self.stats_mode = 1
            # self.stats_x_name = 't'
            # self.stats_y_name = 'm'
            if self.search_algo in [0]:
                self.random_params()
                self.random_world()
            if self.search_algo in [1, 2, 3]:
                self.backup_world()
                if self.search_algo in [3]:
                    self.breadth_count = 0
                if self.search_algo in [1, 3]:
                    self.search_back2 = None
                self.random_params(is_incremental=True)
            elif self.search_algo in [4, 5, 6]:
                self.backup_world()
                self.leaderboard = [{'fitness':float('-inf'), 'world':None} for i in range(self.leaderboard_size)]
                self.automaton.reset()
                self.analyzer.reset()
                self.search_stage = 2  # quick start
                # self.analyzer.trim_segment = 0
                # self.stats_x_name = 's'
        else:
            self.is_auto_center = True
            self.is_auto_load = True

    def finish_search(self):
        if self.search_mode == 0:
            if self.search_algo in [0, 1, 2, 3]:
                # self.update_lineage()
                self.append_found_file_text('\n')
                self.read_found_animals()
            elif self.search_algo in [4, 5, 6]:
                self.append_found_file_leaderboard()
                self.append_found_file_text('\n')
                self.read_found_animals()
            self.search_back = None
            self.search_back2 = None
        self.search_mode = None
        self.search_stage = 0

    def do_search(self):
        global STATUS
        s = 's+' if self.is_search_small else ''
        test_long = self.search_algo in [1] or (self.search_algo in [3] and self.breadth_count == 3)
        test_short = self.search_algo in [2] or (self.search_algo in [3] and self.breadth_count < 3)
        test_vshort = self.search_algo in [0]
        if self.search_mode == +1:
            if self.analyzer.is_empty: self.key_press_internal(s+'w')
            elif self.analyzer.is_full: self.key_press_internal(s+'q')
        elif self.search_mode == -1:
            if self.analyzer.is_empty: self.key_press_internal(s+'a')
            elif self.analyzer.is_full: self.key_press_internal(s+'s')
        elif self.search_mode == 0:
            if self.markers_mode in [1,3,5,7]:
                is_finish = self.analyzer.is_empty or self.analyzer.is_full or \
                        not (self.analyzer.object_num == -1 or 5 <= self.analyzer.object_num <= 10)
            else:
                is_finish = self.analyzer.is_empty or self.analyzer.is_full
            if is_finish:
                self.is_show_search = True
                self.search_stage = 1
                self.search_total += 1
                if self.search_algo in [0]:
                    self.random_params()
                    self.random_world()
                elif self.search_algo in [1, 2, 3]:
                    self.restore_world()
                    self.random_params(is_incremental=True)
                    if self.search_algo in [1]:
                        self.search_back2 = None
                elif self.search_algo in [4, 5, 6]:
                    self.mutate_world_from_leaderboard()
            elif test_long and self.automaton.gen >= 500 and self.search_stage == 1:
                self.is_show_search = True
                self.search_stage = 2
                self.backup_world(i=2, is_reset=False)
            elif (test_vshort and self.automaton.gen >= 250) or \
                 (test_short and self.automaton.gen >= 500) or \
                 (test_long and self.automaton.gen >= 750):
                self.is_show_search = True
                self.search_stage = 1
                self.search_total += 1
                self.search_success += 1
                if test_long:
                    self.restore_world(i=2)
                    self.search_back2 = None
                # self.update_lineage()
                self.append_found_file()
                if self.search_algo in [0]:
                    self.random_params()
                    self.random_world()
                elif self.search_algo in [1]:
                    self.random_params(is_incremental=True)
                    self.backup_world()
                elif self.search_algo in [2]:
                    self.restore_world()
                    self.random_params(is_incremental=True)
                elif self.search_algo in [3]:
                    if self.breadth_count == 3:
                        self.random_params(is_incremental=True)
                        self.backup_world()
                        self.breadth_count = 0
                    else:
                        self.restore_world()
                        self.random_params(is_incremental=True)
                        self.breadth_count += 1
            elif self.search_algo in [4, 5, 6] and self.automaton.gen >= 200 and self.search_stage == 1:
                self.is_show_search = True
                self.search_stage = 2
                self.automaton.reset()
                self.analyzer.reset()
            elif self.search_algo in [4, 5, 6] and self.automaton.gen >= 200 and self.search_stage == 2:
                self.is_show_search = True
                self.search_stage = 1
                self.search_total += 1
                # self.update_lineage()
                self.append_found_file()
                self.put_world_in_leaderboard()
                self.mutate_world_from_leaderboard()
            elif self.automaton.gen % 50 in [0, 5, 10]:
                self.is_show_search = True

            if self.is_show_search:
                if self.search_algo in [0, 1, 2, 3]:
                    STATUS.append("{success} found in {trial} trials ({algo}), saving to {path}".format(success=self.search_success, trial=self.search_total, algo=self.get_value_text('search_algo'), path=self.FOUND_ANIMALS_PATH))
                elif self.search_algo in [4, 5, 6]:
                    STATUS.append("{leader1:.3f}, {leader2:.3f}, {leader3:.3f} leading in {trial} steps ({algo})".format(stat=self.stats_x_name, leader1=self.leaderboard[0]['fitness'], leader2=self.leaderboard[1]['fitness'], leader3=self.leaderboard[2]['fitness'], trial=self.search_total, algo=self.get_value_text('search_algo')))

    def clean_code(self, code):
        if '<' in code:
            return code.split('<')[0]
        else:
            return code

    def update_lineage(self, name='', cname=''):
        new_code = self.random_code(size=6)
        prev_code = self.clean_code(self.world.names['code'])
        if self.search_algo in [0]:
            self.world.names = {'code':new_code, 'name':'', 'cname':'seed:'+self.last_seed}
        elif self.search_algo in [1, 2, 3]:
            self.world.names = {'code':new_code+'<'+prev_code, 'name':'', 'cname':''}
        elif self.search_algo in [4, 5, 6]:
            self.world.names = {'code':new_code+'<'+prev_code, 'name':name, 'cname':cname}

    def append_found_file(self, world=None, newline=',\n'):
        if world is None:
            world = self.world
        A = copy.deepcopy(world)
        A.crop()
        data = A.to_data()
        self.found_animal_data.append(data)
        try:
            with open(self.FOUND_ANIMALS_PATH, 'a+', encoding='utf-8') as file:
                st = json.dumps(data, separators=(',', ':'), ensure_ascii=False) + newline
                file.write(st)
        except IOError as e:
            STATUS.append("I/O error({}): {}".format(e.errno, e.strerror))

    def append_found_file_text(self, text):
        try:
            with open(self.FOUND_ANIMALS_PATH, 'a+', encoding='utf-8') as file:
                file.write(text)
        except IOError as e:
            STATUS.append("I/O error({}): {}".format(e.errno, e.strerror))

    def append_found_file_leaderboard(self):
        # self.append_found_file(world=self.search_back)
        self.append_found_file_text("\"Leaderboard: {algo} where {stat}\",\n".format(stat=self.analyzer.stats_fullname(i=self.stats_x), algo=self.get_value_text('search_algo')))
        for entity in self.leaderboard:
            if entity['world'] is not None:
                self.append_found_file(world=entity['world'])

    def create_window(self):
        self.window = tk.Tk()
        #self.window.tk.call('tk', 'scaling', 2.0)
        #self.window.tk.call('tk', 'scaling', '-displayof', '.', 50)
        self.window.title("Lenia {d}D".format(d=DIM))
        icon_no = np.random.randint(5)+1
        icon = tk.Image("photo", file="resource/icon"+str(icon_no)+".png")
        self.window.call('wm','iconphoto', self.window._w, icon)
        self.window.bind('<Key>', self.key_press_event)
        self.frame = tk.Frame(self.window, width=SIZEX*PIXEL, height=SIZEY*PIXEL)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=SIZEX*PIXEL, height=SIZEY*PIXEL)
        self.canvas.place(x=-1, y=-1)
        self.panel1 = self.create_panel(0, 0, SIZEX, SIZEY)
        # self.panel2 = self.create_panel(1, 0)
        # self.panel3 = self.create_panel(0, 1)
        # self.panel4 = self.create_panel(1, 1)
        self.info_bar = tk.Label(self.window)
        self.info_bar.pack()

    def create_panel(self, c, r, w, h):
        buffer = np.uint8(np.zeros((h*PIXEL, w*PIXEL)))
        img = PIL.Image.frombuffer('P', (w*PIXEL,h*PIXEL), buffer, 'raw', 'P', 0, 1)
        photo = PIL.ImageTk.PhotoImage(image=img)
        return self.canvas.create_image(c*PIXEL, r*PIXEL, image=photo, anchor=tk.NW)

    def create_colormap(self, colors, is_marker_w=True):
        nval = 253
        ncol = colors.shape[0]
        colors = np.vstack((colors, np.asarray([[0,0,0]])))
        v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
        i = np.asarray(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
        k = v / (nval-1) * (ncol-1)  # interpolate between 0 .. ncol-1
        k1 = k.astype(int)
        c1, c2 = colors[k1,i], colors[k1+1,i]
        c = (k-k1) * (c2-c1) + c1  # interpolate between c1 .. c2
        return np.rint(c / 8 * 255).astype(int).tolist() + (self.MARKER_COLORS_W if is_marker_w else self.MARKER_COLORS_B)

    def create_colormap_flat(self, c0, c1, is_marker_w=True):
        nval = 253
        x = np.linspace(1, 1, nval-1)
        c = np.asarray([c1 * x, c1 * x, c1 * x])
        c = np.hstack((np.asarray([[c0], [c0], [c0]]), c))
        c = np.clip(c, 0, 255)
        return np.rint(c.flatten('F')).astype(int).tolist() + (self.MARKER_COLORS_W if is_marker_w else self.MARKER_COLORS_B)

    def create_colormap_turbo(self, seq='rgb', is_marker_w=True):
        ''' https://observablehq.com/@mbostock/turbo '''
        nval = 253
        x = np.linspace(0, 1, nval)
        s = {}
        s['r'] = 34.61 + x * (1172.33 - x * (10793.56 - x * (33300.12 - x * (38394.49 - x * 14825.05))))
        s['g'] = 23.31 + x * (557.33 + x * (1225.33 - x * (3574.96 - x * (1073.77 + x * 707.56))))
        s['b'] = 27.2 + x * (3211.1 - x * (15327.97 - x * (27814 - x * (22569.18 - x * 6838.66))))
        c = np.asarray([s[seq[0]], s[seq[1]], s[seq[2]]])
        c = np.clip(c, 0, 255)
        return np.rint(c.flatten('F')).astype(int).tolist() + (self.MARKER_COLORS_W if is_marker_w else self.MARKER_COLORS_B)

    def is_show_rgb(self):
        return (self.show_what==0 and CN>1) or (self.show_what in [1,2,3] and self.show_group==1 and CN>1)

    def get_color(self, n):
        if n is None:
            return None
        if self.is_show_rgb():
            colormap = self.colormaps[self.colormap_id]
            return tuple(colormap[n*3+i] for i in range(3))
        else:
            return n

    def show_which_channels(self, A):
        '''
        A = np.asarray(A)
        a,b,c = 6,3,0  # 8,0,0  6,3,0  7,2,0
        m = np.asarray([[a,b,c],[c,a,b],[b,c,a]]).T / 8  #YCM
        b = np.asarray([0,1,0]).reshape(3,1,1) / 8
        # m = np.asarray([[8,0,0],[0,8,0],[0,0,8]]).T / 8  #RGB
        # b = np.asarray([0,0,3]).reshape(3,1,1) / 8
        # A = np.einsum('cv,vij->cij', m, A) + b
        A = np.tensordot(m, A, axes=1) + b
        A = [A[0], A[1], A[2]]
        '''
        if self.is_show_rgb():
            r = np.zeros(A[0].shape)
            g = np.zeros(A[0].shape)
            b = np.zeros(A[0].shape)
            for c in CHANNEL:
                m = self.channelmaps[self.channel_group][(c + self.channel_shift) % CN]
                # r = np.maximum(r, A[c] * m[0])
                r += A[c] * m[0] * 3/CN
                g += A[c] * m[1] * 3/CN
                b += A[c] * m[2] * 3/CN
            r += self.channelbg[self.channel_group][0] / (DIM-1)
            g += self.channelbg[self.channel_group][1] / (DIM-1)
            b += self.channelbg[self.channel_group][2] / (DIM-1)
            A = [r, g, b]
        return A

    def show_which_channels_name(self):
        if self.is_show_rgb():
            group = ["RGBK","RBGK","OTVK","OVTK","WEEK"][self.channel_group]
            st = []
            for c in CHANNEL:
                color = group[(c + self.channel_shift) % len(group)]
                st.append('{c}:{color}'.format(c=c, color=color))
            return ','.join(st)
        else:
            return ''

    def update_window(self, show_arr=None, is_reimage=True):
        if is_reimage:
            # if self.show_what==6:
            #     xgrad, ygrad = np.gradient(self.automaton.potential)
            #     grad = np.sqrt(xgrad**2 + ygrad**2) * self.world.model['R'] * 1.5
            #     #print(np.min(grad), np.max(grad))

            if show_arr is not None:
                self.draw_world(show_arr, 0, 1)
            elif self.stats_mode in [0, 1, 2, 5]:
                change_range = 1 if self.automaton.soft_clip_level == 0 else 1.4
                field_lo = -1 if not self.automaton.is_arita_mode else 0
                if self.show_what==0: self.draw_world(self.show_which_channels(self.world.cells), 0, 1, is_shift=True, is_higher_zero=True, markers=['world','marks','scale','grid','colormap','params'])
                elif self.show_what==1: self.draw_kernel(self.automaton.potential, 0, 0.5, vmax_m=2, is_shift=True, is_higher_zero=True, markers=['marks','scale','grid','colormap','params'])
                elif self.show_what==2: self.draw_kernel(self.automaton.field, field_lo, 0.3, is_shift=True, markers=['marks','scale','grid','colormap','params'])
                elif self.show_what==3: self.draw_kernel(self.automaton.kernel, 0, 1, markers=['scale','fixgrid','colormap','params'])
                elif self.show_what==4: self.draw_world(self.analyzer.object_map / self.analyzer.object_num - self.analyzer.peak_mask*1, 0, 1, is_shift=True, markers=['marks','scale','grid','colormap','params'])
                # elif self.show_what==4: self.draw_world(self.automaton.fftshift(np.log(np.abs(self.automaton.world_FFT))), 0, 5, markers=['colormap','params'])  #-10, 10
                # elif self.show_what==5: self.draw_world(self.automaton.fftshift(np.log(np.abs(self.automaton.potential_FFT))), -20, 5, markers=['colormap','params'])  #-40, 10
                # elif self.show_what==6: self.draw_world(grad, 0, 1, is_shift=True, markers=['marks','scale','grid','colormap','params'])
                # elif self.show_what==7: self.draw_world(self.automaton.change, -change_range, change_range, is_shift=True, markers=['marks','scale','grid','colormap','params'])
            elif self.stats_mode in [3, 4]:
                self.draw_black()
                self.draw_stats(is_current_series=self.stats_mode in [1, 2, 3], is_small=self.stats_mode in [1])
            elif self.stats_mode in [6]:
                self.draw_recurrence()

            elif self.stats_mode in [5]:
                self.draw_psd(is_welch=True)

            if self.recorder.is_recording and self.is_run:
                self.recorder.record_frame(self.img)
            if self.is_save_image:
                if not os.path.exists(self.SAVE_ROOT):
                    os.makedirs(self.SAVE_ROOT)
                self.recorder.save_image(self.img, filename=os.path.join(self.SAVE_ROOT, str(self.file_seq)))
                self.is_save_image = False

        photo1 = PIL.ImageTk.PhotoImage(image=self.img)
        # photo = tk.PhotoImage(width=SIZEX, height=SIZEY)
        self.canvas.itemconfig(self.panel1, image=photo1)
        self.window.update()

    def normalize(self, v, vmin, vmax, is_square=False, vmin2=0, vmax2=0):
        if not is_square:
            return (v-vmin) / (vmax-vmin)
        else:
            return (v-vmin) / max(vmax-vmin, vmax2-vmin2)

    def draw_cell_borders(self, buffer):
        # zero = np.uint8(np.clip(self.normalize(0, vmin, vmax), 0, 1) * 252)
        zero = 0
        for b in buffer:
            for i in range(PIXEL_BORDER):
                b[i::PIXEL, :] = zero;  b[:, i::PIXEL] = zero

    def get_image(self, buffer):
        y, x = buffer[0].shape
        buffer = [np.repeat(b, PIXEL, axis=0) for b in buffer]
        buffer = [np.repeat(b, PIXEL, axis=1) for b in buffer]
        self.draw_cell_borders(buffer)
        if self.is_show_rgb():
            buffer = np.dstack(buffer)
            return PIL.Image.frombuffer('RGB', (x*PIXEL,y*PIXEL), buffer, 'raw', 'RGB', 0, 1)
        else:
            return PIL.Image.frombuffer('P', (x*PIXEL,y*PIXEL), buffer[0], 'raw', 'P', 0, 1)

    def shift_img(self, img, dx, dy, is_rotate=True):
        sx, sy = img.size
        if dx != 0:
            if is_rotate: part1 = img.crop((0, 0, dx, sy))
            part2 = img.crop((dx, 0, sx, sy))
            img.paste(part2, (0, 0, sx-dx, sy))
            if is_rotate: img.paste(part1, (sx-dx, 0, sx, sy))
        if dy != 0:
            if is_rotate: part1 = img.crop((0, 0, sx, dy))
            part2 = img.crop((0, dy, sx, sy))
            img.paste(part2, (0, 0, sx, sy-dy))
            if is_rotate: img.paste(part1, (0, sy-dy, sx, sy))

    def get_kernel_array(self, A, vmin=0, vmax=1, vmax_m=None, **kwargs):
        '''
        if self.show_group == 0:
            A2 = sum(A)
        '''
        if self.show_group == 0:
            #A2 = sum(A) / len(A)
            sum_Ah = sum(A0 * p['h'] for A0, p in zip(A, self.world.params))
            sum_h = sum(p['h'] for p in self.world.params)
            A2 = sum_Ah / sum_h
        elif self.show_group == 1:
            A2 = []
            for c in CHANNEL:
                sum_Ah = sum(A0 * p['h'] for A0, p in zip(A, self.world.params) if p['c1']==c)
                sum_h = sum(p['h'] for p in self.world.params if p['c1']==c)
                A2.append(sum_Ah / sum_h)
                #Ah = [A0 for A0, p in zip(A, self.world.params) if p['c1']==c]
                #A2.append(sum(Ah) / len(Ah))
            A2 = self.show_which_channels(A2)
        elif self.show_group == 2:
            A2 = A[self.show_kernel]
            if vmax_m is not None:
                vmax = vmax_m*self.world.params[self.show_kernel]['m']
        return A2

    def draw_kernel(self, A, vmin=0, vmax=1, vmax_m=None, **kwargs):
        A2 = self.get_kernel_array(A, vmin, vmax, vmax_m, **kwargs)
        self.draw_world(A2, vmin, vmax, **kwargs)

    def draw_world(self, A, vmin=0, vmax=1, is_shift=False, is_higher_zero=False, markers=[]):
        R = self.world.model['R']

        if type(A) not in [list]:
            A = [A]
        if 'marks' in markers and self.markers_mode in [1,3,5,7]:
            A = [A0.copy() for A0 in A]
            mask = self.analyzer.object_border + self.analyzer.peak_mask
            label = self.analyzer.object_map / self.analyzer.object_num * (vmax-vmin) + vmin
            if self.is_show_rgb():
                ones = np.ones(label.shape)
                hsv = np.dstack([label, ones, ones])
                rgb = skimage.color.hsv2rgb(hsv)
                np.putmask(A[0], mask, rgb[...,0])
                np.putmask(A[1], mask, rgb[...,1])
                np.putmask(A[2], mask, rgb[...,2])
            else:
                for A0 in A:
                    np.putmask(A0, mask, label)

        is_xy = self.stats_x_name in ['x'] and self.stats_y_name in ['y'] and self.stats_mode in [2]
        axes = tuple(reversed(range(DIM)))
        if is_shift and not self.is_auto_center:
            shift = self.analyzer.total_shift_idx #if 'world' in markers else self.analyzer.total_shift_idx - self.analyzer.last_shift_idx 
            A = [np.roll(A0, shift.astype(int), axes) for A0 in A]
            # A = scipy.ndimage.shift(A, self.analyzer.total_shift_idx, order=0, mode='wrap')
        if is_higher_zero and self.automaton.soft_clip_level > 0 and vmin==0:
            vmin = min([np.amin(A0) for A0 in A])

        angle_shift = 0
        if DIM == 2 and is_shift:
            if self.auto_rotate_mode in [1]:
                angle_shift = -self.analyzer.m_angle/360 - 0.25
            elif self.auto_rotate_mode in [2]:
                angle_shift = self.analyzer.symm_angle/2/np.pi
            elif self.auto_rotate_mode in [3]:
                angle_shift = self.samp_rotate*self.automaton.time/360

        if self.polar_mode in [0,1]:
            if DIM > 2:
                if self.is_show_slice:
                    for d in range(DIM-2):
                        A = [A0[self.z_slices[d]] for A0 in A]
                else:
                    for d in range(DIM-3):
                        A = [A0[self.z_slices[d]] for A0 in A]
                    #auto-spin
                    #A = scipy.ndimage.rotate(A, (self.automaton.gen*2) % 360, axes=(X_AXIS, self.z_axis), reshape=False, order=1, mode='wrap')
                    A = [(A0 * self.automaton.Z_depth).sum(axis=0) for A0 in A]

            if self.is_show_rgb():
                buffer = [np.uint8(np.clip(self.normalize(A0, vmin, vmax), 0, 1) * 252) for A0 in A]
                self.draw_grid_rgb(buffer, markers, is_fixed='fixgrid' in markers)
                self.img = self.get_image(buffer)
            else:
                buffer = np.uint8(np.clip(self.normalize(A[0], vmin, vmax), 0, 1) * 252)  # .copy(order='C')
                self.draw_grid(buffer, markers, is_fixed='fixgrid' in markers)
                # self.draw_params(buffer, markers)
                self.img = self.get_image([buffer])
            self.draw_marks(markers)
            self.draw_symmetry(markers)
            if PIXEL > 1 and self.is_auto_center and is_shift and self.analyzer.m_center is not None:
                m1 = self.analyzer.m_center * R * PIXEL
                self.shift_img(self.img, int(m1[0]), int(m1[1]), is_rotate=False)
            if is_xy:
                self.draw_stats(is_draw_text=False, is_current_series=self.stats_mode in [1, 2, 3], is_small=self.stats_mode in [1])
            if angle_shift != 0:
                self.img = self.img.rotate(-angle_shift*360, resample=PIL.Image.NEAREST, expand=False)
                # samp_rotate = OG:96, D7:-8, D8:-6, D9:-5.4
            self.draw_symmetry_title(markers)
            self.draw_legend(markers, vmin, vmax)
            #if self.stats_mode in [1]:
            #    self.draw_histo(A, vmin, vmax)
        elif self.polar_mode in [2,3,4] and self.analyzer.polar_array is not None:
            if self.polar_mode in [2] and self.analyzer.is_calc_symmetry:
                A2 = self.analyzer.polar_array
                X = self.analyzer.polar_TH
                Y = self.analyzer.polar_R.reshape((-1, 1))
                k = self.analyzer.symm_sides
                if k > 0:
                    X_max = np.amax(X)
                    Y_max = np.amax(Y)
                    # A2[:SIZER, -2:] = Y[:] / Y_max
                    A2[:2, :] = X[:] / X_max
                    p = int(np.ceil(SIZETH / k))
                    X_intp = np.interp(np.linspace(0, 1, p*k), np.linspace(0, 1, SIZETH), X)  # np.linspace(0, 1, SIZEX)+a/2/np.pi , +a/2/np.pi-int(a/2/np.pi*SIZEX)/SIZEX
                    X_stack = np.asarray(np.hsplit(X_intp, k))
                    for i in range(k):
                        A2[3+i,:p] = X_stack[i,:] / X_max
                buffer = np.uint8(np.clip(self.normalize(A2, vmin, vmax), 0, 1) * 252)
            elif self.polar_mode in [3] and self.analyzer.polar_array is not None and self.analyzer.series_TH is not None:
                A2 = np.zeros(self.analyzer.polar_array.shape)
                if len(self.analyzer.series_TH) > 0:
                    X = np.asarray(self.analyzer.series_TH)
                    Y = np.asarray(self.analyzer.series_R).transpose()
                    X_len = min(X.shape[0], SIZER-1)
                    Y_len = min(Y.shape[1], SIZETH)
                    X = X[-X_len:, :SIZETH]
                    Y = Y[:SIZER, -Y_len:]
                    A2[MIDY+X_len-1:MIDY-1:-1, :SIZETH] = X / X.max()
                    A2[:SIZER, :Y_len] = Y / Y.max()
                buffer = np.uint8(np.clip(self.normalize(A2, vmin, vmax), 0, 1) * 252)
            elif self.polar_mode in [4] and self.analyzer.polar_density is not None:
                A2 = np.vstack((
                    self.analyzer.polar_density / np.amax(self.analyzer.polar_density),
                    self.analyzer.rotate_wsum / np.amax(self.analyzer.rotate_wsum)))
                A2[:2, :] = self.analyzer.density_sum[:] / np.amax(self.analyzer.density_sum)
                buffer = np.uint8(np.clip(self.normalize(A2, vmin, vmax), 0, 1) * 252)
                buffer = np.repeat(buffer, 2, axis=1)
            self.img = self.get_image([buffer])
            self.draw_marks(markers)
            self.draw_symmetry(markers)
            if self.polar_mode in [2] and angle_shift != 0:
                dx = int((-angle_shift % 1)*SIZETH*PIXEL)
                self.shift_img(self.img, dx, 0, is_rotate=True)
            self.draw_symmetry_title(markers)
            self.draw_legend(markers, vmin, vmax)

        if not self.is_show_rgb():
            self.img.putpalette(self.colormaps[self.colormap_id])
        if self.stats_mode in [1, 2]:
            self.draw_stats(is_draw_line=not is_xy, is_current_series=self.stats_mode in [1, 2, 3], is_small=self.stats_mode in [1])

    def draw_title(self, draw, line, title, color=255):
        title_w, title_h = draw.textsize(title)
        title_x, title_y = MIDX*PIXEL-title_w//2, line*12+7
        draw.text((title_x,title_y), title, fill=self.get_color(color), font=self.font)

    # def draw_histo(self, A, vmin, vmax):
        # draw = PIL.ImageDraw.Draw(self.img)
        # HWIDTH = 1
        # hist, _ = np.histogram(A, bins=SIZEX//HWIDTH, range=(vmin,vmax))  #, density=True)
        # #print(vmin, vmax, A.min(), A.max())
        # for i in range(hist.shape[0]):
            # h = hist[i]
            # y = h  #(h*SIZEY).astype(int)
            # draw.rectangle([(i*HWIDTH*PIXEL,SIZEY*PIXEL),((i+1)*HWIDTH*PIXEL-1,(SIZEY-y)*PIXEL)], fill=self.get_color(254))
        # del draw

    def draw_black(self):
        isize = (SIZEX*PIXEL,SIZEY*PIXEL)
        asize = (SIZEY*PIXEL,SIZEX*PIXEL)
        if self.is_show_rgb():
            self.img = PIL.Image.frombuffer('RGB', isize, np.zeros(asize), 'raw', 'RGB', 0, 1)
        else:
            self.img = PIL.Image.frombuffer('L', isize, np.zeros(asize), 'raw', 'L', 0, 1)

    def draw_grid(self, buffer, markers=[], is_fixed=False):
        if not ( ('grid' in markers or 'fixgrid' in markers) and self.markers_mode in [0,1,2,3] ):
            return
        R = self.world.model['R']
        n = R // 40 if R >= 15 else -1
        for i in range(-n, n+1):
            sx, sy = 0, 0
            if self.is_auto_center and not is_fixed:
                sx, sy, *_ = self.analyzer.total_shift_idx.astype(int)
            grid = buffer[(MIDY - sy + i) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R];  grid[grid==0] = 253
            grid = buffer[(MIDY - sy) % R:SIZEY:R, (MIDX - sx + i) % R:SIZEX:R];  grid[grid==0] = 253

    def draw_grid_rgb(self, buffer, markers=[], is_fixed=False):
        if not ( ('grid' in markers or 'fixgrid' in markers) and self.markers_mode in [0,1,2,3] ):
            return
        R = self.world.model['R']
        sx, sy = 0, 0
        if self.is_auto_center and not is_fixed:
            sx, sy, *_ = self.analyzer.total_shift_idx.astype(int)
        for b in buffer:
            b[(MIDY - sy) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R] = 0x5F
        # cell_grid = [c[(MIDY - sy) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R] for c in self.world.cells]
        # cell_grid_all = sum(cell_grid)
        # grid = [b[(MIDY - sy) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R] for b in buffer]
        # for g in grid:
        #     g[cell_grid_all<EPSILON] = 0x7F

    '''
    def draw_params(self, buffer, markers=[]):
        if self.is_draw_params and 'params' in markers and self.markers_mode in [0,2,4,6] and self.polar_mode in [0,1]:
            p = self.world.params[self.show_kernel]
            s = 12
            b = p['b'].copy()
            b += [0] * (3-len(b))
            r = range(s+1)
            mx, my = 2, 2
            for (d1, d2, d3) in [(d1, d2, d3) for d1 in r for d2 in r for d3 in r  if d1==s or d2==s or d3==s]:
                x, y = self.cube_xy(d1, d2, d3, s)
                if len(b) > 3: c = 253
                elif b[0]==Fraction(d1,s) and b[1]==Fraction(d2,s) and b[2]==Fraction(d3,s): c = 255
                elif (d1, d2, d3).count(s) >= 2: c = 254
                else: c = 253
                buffer[y+my, x*2+mx:x*2+2+mx] = c  # y+my:y+2+my
            mx, my = 2, SIZEY - 53
            for pm in range(51):
                buffer[51-pm+my, 0+mx] = 253
            for ps in range(61):
                buffer[51-0+my, ps+mx] = 253
            pm = int(np.floor(p['m'] / 0.01))
            ps = int(np.floor(p['s'] / 0.002))
            if 0<=pm<=50 and 0<=ps<=60:
                buffer[51-pm+my, ps+mx] = 255
    '''

    def draw_marks(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.model[k] for k in ('R', 'T')]
        midpoint = np.asarray(MID)
        dd = np.asarray([2]*DIM)
        if 'marks' in markers and self.markers_mode in [1,3,5,7] and self.polar_mode in [0,1] and R > 2 and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
            shift = self.analyzer.total_shift_idx if not self.is_auto_center else np.zeros(DIM)
            m0 = self.analyzer.m_last_center * R + midpoint + shift - self.analyzer.last_shift_idx
            m1 = self.analyzer.m_center * R + midpoint + shift
            ms = m1 % np.asarray(SIZE) - m1
            m2, m3 = [m0 + (m1 - m0) * n * T for n in [1, 2]]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    D = [i,j] + [0]*(DIM-2)
                    adj = np.asarray([d*size for d, size in zip(D, SIZE)]) + ms
                    p1 = (m0+adj) * PIXEL
                    p2 = (m3+adj) * PIXEL
                    draw.line([p1[0], p1[1], p2[0], p2[1]], fill=self.get_color(254), width=1)
                    for (m,c) in [(m0,254),(m1,255),(m2,255),(m3,255)]:
                        p1 = (m+adj) * PIXEL - dd
                        p2 = (m+adj) * PIXEL + dd
                        draw.ellipse([p1[0], p1[1], p2[0], p2[1]], fill=self.get_color(c))
        del draw

    def draw_symmetry_title(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.model[k] for k in ('R', 'T')]
        if self.analyzer.is_calc_symmetry and 'scale' in markers and self.markers_mode in [1,3,5,7] and R > 2:
            k = self.analyzer.symm_sides
            self.draw_title(draw, 0, "symmetry: {k} ({name})".format(k=k, name=self.POLYGON_NAME[k] if k <= 10 else self.POLYGON_NAME[0]))
        del draw

    def draw_symmetry(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.model[k] for k in ('R', 'T')]
        midpoint = np.asarray([MIDX, MIDY])
        dd = np.asarray([1, 1]) * 2
        if self.analyzer.is_calc_symmetry and 'marks' in markers and self.markers_mode in [0,1,2,3] and R > 2 and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
            is_draw_radial = self.polar_mode in [1] or (self.polar_mode in [0] and self.auto_rotate_mode in [2])
            shift = self.analyzer.total_shift_idx if not self.is_auto_center else np.zeros(2)
            m1 = self.analyzer.m_center * R + midpoint + shift
            m1 %= np.asarray([SIZEX, SIZEY])
            if self.auto_rotate_mode in [3]:
                k = self.samp_sides
                a = np.radians(self.samp_rotate*self.automaton.time)
            else:
                k = self.analyzer.symm_sides
                a = self.analyzer.symm_angle
            if self.analyzer.sides_vec is not None:
                kk = self.analyzer.sides_vec
                aa = self.analyzer.angle_vec
                ww = self.analyzer.rotate_vec * T
                if is_draw_radial or self.polar_mode in [2]:
                    #draw symmetry lines (radial or vertical)
                    if k > 1:
                        for i in range(k):
                            if is_draw_radial:
                                #radial
                                angle = 2*np.pi * i / k + a
                                d1 = np.asarray([np.sin(angle), np.cos(angle)])*max(SIZEX, SIZEY)
                                draw.line(tuple(m1*PIXEL) + tuple((m1-d1)*PIXEL), fill=self.get_color(254), width=1)
                            elif self.polar_mode in [2]:
                                #vertical
                                x = SIZETH * ((i / k - a/2/np.pi + 0.5) % 1)
                                draw.line((x*PIXEL, 0*PIXEL, x*PIXEL, SIZEY*PIXEL), fill=self.get_color(254), width=1)
                elif self.polar_mode in [4]:
                    #draw vertical lines per 5 sides in symmetry
                    for i in range(1, SIZEF, 5):
                        draw.line((i*2*PIXEL, 0*PIXEL, i*2*PIXEL, SIZEY*PIXEL), fill=self.get_color(254), width=1)
                        x0, y0 = i*2*PIXEL+2, MIDY*PIXEL
                        draw.text((x0,y0), str(i), fill=self.get_color(255), font=self.font)
                if self.polar_mode in [2,3,4]:
                    #draw horizontal line separate upper and lower panels
                    draw.line((0*PIXEL, SIZER*PIXEL, SIZEX*PIXEL, SIZER*PIXEL), fill=self.get_color(254), width=1)
                #draw rotational angle circles and rotational speed lines (circumferential or horizontal)
                for r in range(kk.size):
                    if kk[r] > 1:
                        if is_draw_radial:
                            #circumferential
                            c = 255 if kk[r] == k else 254
                            for i in range(kk[r]):
                                angle = 2*np.pi * i / kk[r] + aa[r]
                                d1 = np.asarray([np.sin(angle), np.cos(angle)])*(SIZER-r)
                                # d2 = np.asarray([np.sin(angle+ww[r]), np.cos(angle+ww[r])])*(SIZER-r)
                                # draw.line(tuple((m1-d1)*PIXEL) + tuple((m1-d2)*PIXEL), fill=self.get_color(c), width=d)
                                th1 = 270 - np.degrees(angle)
                                th2 = th1 - np.degrees(ww[r])
                                if th1 > th2: th1,th2 = th2,th1
                                draw.arc(tuple((m1-SIZER+r)*PIXEL) + tuple((m1+SIZER-r)*PIXEL), th1, th2, fill=self.get_color(c), width=1)
                                draw.ellipse(tuple((m1-d1)*PIXEL-dd) + tuple((m1-d1)*PIXEL+dd), fill=self.get_color(c))
                        elif self.polar_mode in [2]:
                            #horizontal
                            c = 255 if kk[r] == k else 254
                            for i in range(kk[r]):
                                x = SIZETH * ((i / kk[r] - aa[r]/2/np.pi + 0.5) % 1)
                                draw.line((x*PIXEL, r*PIXEL, (x-ww[r]/2/np.pi*SIZETH)*PIXEL, r*PIXEL), fill=self.get_color(c), width=1)
                                draw.ellipse((x*PIXEL-2, r*PIXEL-2, x*PIXEL+2, r*PIXEL+2), fill=self.get_color(c))
                        elif self.polar_mode in [4]:
                            #horizontal
                            c = 255
                            x = (kk[r]+1) * PIXEL//2
                            draw.line((x*PIXEL, r*PIXEL, (x-ww[r]/2/np.pi*SIZETH)*PIXEL, r*PIXEL), fill=self.get_color(c), width=1)
                            draw.ellipse((x*PIXEL-2, r*PIXEL-2, x*PIXEL+2, r*PIXEL+2), fill=self.get_color(c))
        del draw

    def draw_legend(self, markers=[], vmin=0, vmax=1):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.model[k] for k in ('R', 'T')]
        midpoint = np.asarray([MIDX, MIDY])
        dd = np.asarray([1, 1]) * 2
        if 'marks' in markers and self.markers_mode in [1,3,5,7] and self.polar_mode in [0,1]:
            #draw speed legend
            x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-35
            draw.line([(x0-90,y0),(x0,y0)], fill=self.get_color(254), width=1)
            for (m,c) in [(0,254),(-10,255),(-50,255),(-90,255)]:
                draw.ellipse(tuple((x0+m,y0)-dd) + tuple((x0+m,y0)+dd), fill=self.get_color(c))
            draw.text((x0-95,y0-20), '2s', fill=self.get_color(255), font=self.font)
            draw.text((x0-55,y0-20), '1s', fill=self.get_color(255), font=self.font)
        if 'scale' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1]:
            #draw scale bar
            x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-20
            draw.text((x0+10,y0), '1mm', fill=self.get_color(255), font=self.font)
            draw.rectangle([(x0-R*PIXEL,y0+3),(x0,y0+7)], fill=self.get_color(255))
        if 'colormap' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1]:
            #draw colormap
            ncol = 256 if self.is_show_rgb() else 253
            bar_w = 3*3 if self.is_show_rgb() else 5
            x0, y0 = SIZEX*PIXEL-20, SIZEY*PIXEL-70
            x1, y1 = SIZEX*PIXEL-21+bar_w, 20
            dy = (y1-y0)/ncol
            if self.is_show_rgb():
                x0 = x1-CN*3+1
                for c in CHANNEL:
                    m = self.channelmaps[self.channel_group][(c + self.channel_shift) % CN]
                    for val in range(ncol):
                        color = (m * val).astype(int).tolist()
                        draw.rectangle([(x0+c*3,y0+dy*val),(x0+(c+1)*3-1,y0+dy*(val+1))], fill=tuple(color))
            else:
                for c in range(ncol):
                    draw.rectangle([(x0,y0+dy*c),(x1,y0+dy*(c+1))], fill=self.get_color(c))
            draw.rectangle([(x0-1,y0+1),(x1+1,y1-1)], outline=self.get_color(254))
            draw.text((x0-25,y0-5), "{:.1f}".format(vmin), fill=self.get_color(255), font=self.font)
            draw.text((x0-25,(y1+y0)//2-5), "{:.1f}".format((vmin+vmax)/2), fill=self.get_color(255), font=self.font)
            draw.text((x0-25,y1-5), "{:.1f}".format(vmax), fill=self.get_color(255), font=self.font)
        del draw

    def cube_xy(self, d1, d2, d3, s):
        return (s + d1 - d3), (2*s + d1 - 2*d2 + d3)

    def draw_stats(self, is_current_series=True, is_small=True, is_draw_line=True, is_draw_text=True):
        R = self.world.model['R']
        draw = PIL.ImageDraw.Draw(self.img)
        series = self.analyzer.series
        current = self.analyzer.current
        is_square = self.stats_x_name in ['x', 'y'] and self.stats_y_name in ['x', 'y']
        is_xy = self.stats_x_name in ['x'] and self.stats_y_name in ['y'] and self.stats_mode in [2]
        if series != [] and is_current_series:
            series = [series[-1]]
        if series != [] and series != [[]]:
            X = [np.asarray([val[self.stats_x] for val in seg]) for seg in series if len(seg)>0]
            Y = [np.asarray([val[self.stats_y] for val in seg]) for seg in series if len(seg)>0]
            S = [seg[0][1] for seg in series if len(seg)>0]
            M = [seg[0][0] for seg in series if len(seg)>0]
            # if name_x in ['n', 't']: X = [seg - seg.min() for seg in X]
            # if name_y in ['n', 't']: Y = [seg - seg.min() for seg in Y]
            xmin, xmax = min(seg.min() for seg in X if seg.size>0), max(seg.max() for seg in X if seg.size>0)
            ymin, ymax = min(seg.min() for seg in Y if seg.size>0), max(seg.max() for seg in Y if seg.size>0)
            smin, smax = min(S), max(S)
            mmin, mmax = min(M), max(M)
            # xmean, ymean = (xmax+xmin) / 2, (ymax+ymin) / 2
            # if xmax-xmin < 0.01: xmin, xmax = xmean-0.01, xmean+0.01
            # if ymax-ymin < 0.01: ymin, ymax = ymean-0.01, ymean+0.01
            # if name_x in ['m_a']:
                # mass = [np.asarray([val[4] for val in seg]) for seg in series if len(seg)>0]
                # massmax = max(seg.max() for seg in mass if seg.size>0)
                # if name_x in ['m_a']:
                    # xmin, xmax = min(xmin, -massmax/32), max(xmax, massmax/32)
            title_st_x = "X: {name} ({min:.3f}-{max:.3f}) {val:.3f}".format(name=self.stats_x_name, min=xmin, max=xmax, val=current[self.stats_x])
            title_st_y = "Y: {name} ({min:.3f}-{max:.3f}) {val:.3f}".format(name=self.stats_y_name, min=ymin, max=ymax, val=current[self.stats_y])
            if is_small:
                xmax = (xmax - xmin) * 4 + xmin
                ymax = (ymax - ymin) * 4 + ymin
                y_shift = 32
                title_x, title_y = 5, SIZEY*PIXEL - 32
            else:
                if is_xy:
                    xmax = ymax = R * 2
                y_shift = 0
                title_x, title_y = 5, 5
            if is_draw_text:
                draw.text((title_x,title_y), title_st_x, fill=self.get_color(255), font=self.font)
                draw.text((title_x,title_y+12), title_st_y, fill=self.get_color(255), font=self.font)
            if is_draw_line:
                if not is_current_series:
                    C = list(reversed([194 // 2**i + 61 for i in range(len(X))]))
                else:
                    C = [255] * len(X)
                ds = 0.0001 if self.is_search_small else 0.001
                dm = 0.001 if self.is_search_small else 0.01
                for x, y, m, s, c in zip(X, Y, M, S, C):
                    is_valid = not np.isnan(x[0])
                    if self.is_group_params:
                        xmin, xmax = x.min(), x.max()
                        ymin, ymax = y.min(), y.max()
                        x, y = self.normalize(x, xmin, xmax), self.normalize(y, ymin, ymax)
                        s, m = self.normalize(s, smin, smax+ds), self.normalize(m, mmin, mmax+dm)
                        x, x0, x1 = [(a * ds/(smax-smin+ds) + s) * (SIZEX*PIXEL - 10) + 5 for a in [x, 0, 1]]
                        y, y0, y1 = [(1 - a * dm/(mmax-mmin+dm) - m) * (SIZEY*PIXEL - 10) + 5 for a in [y, 0, 1]]
                        draw.rectangle([(x0,y0),(x1,y1)], outline=self.get_color(c), fill=None if is_valid else c)
                    else:
                        if is_xy and self.analyzer.m_center is not None:
                            midpoint = np.asarray(MID)
                            shift = self.analyzer.total_shift_idx + self.analyzer.m_center * R if self.is_auto_center else np.zeros(DIM)
                            m1 = midpoint - shift
                            x_pane = np.floor((m1[0] + x) / SIZEX)
                            y_pane = np.floor((m1[1] - y) / SIZEY)
                            x = (m1[0] + x) % SIZEX * PIXEL
                            y = (m1[1] - y) % SIZEY * PIXEL
                            x = [x0 if pane==pane2 else None for x0, pane, pane2 in zip(x, x_pane, np.roll(x_pane, 1))]
                            y = [y0 if pane==pane2 else None for y0, pane, pane2 in zip(y, y_pane, np.roll(y_pane, 1))]
                        else:
                            x = self.normalize(x, xmin, xmax, is_square, ymin, ymax) * (SIZEX*PIXEL - 10) + 5
                            y = (1-self.normalize(y, ymin, ymax, is_square, xmin, xmax)) * (SIZEY*PIXEL - 10) + 5 - y_shift
                    if is_valid:
                        if None in x or None in y:
                            for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:]):
                                if not (x0 is None or y0 is None or x1 is None or y1 is None):
                                    draw.line([x0, y0, x1, y1], fill=self.get_color(c), width=1)
                        else:
                            draw.line(list(zip(x, y)), fill=self.get_color(c), width=1)
        del draw

    def draw_psd(self, is_welch=True):
        draw = PIL.ImageDraw.Draw(self.img)
        T = self.world.model['T']
        series = self.analyzer.series
        if self.analyzer.is_calc_psd and self.analyzer.psd_freq is not None:
            self.draw_title(draw, 1, 'periodogram (Welch)' if is_welch else 'periodogram')
            freq = self.analyzer.psd_freq
            xmin, xmax = freq.min(), freq.max()
            self.analyzer.period = 1 / freq[np.argmax(self.analyzer.psd1)]
            self.analyzer.period_gen = self.analyzer.period * T
            for (n, psd, name) in zip([0,1], [self.analyzer.psd2, self.analyzer.psd1], [self.stats_y_name, self.stats_x_name]):
                if psd is not None and psd.shape[0] > 0:
                    #if len(psd.shape) > 1: psd = psd.max(axis=1)
                    c = 254 if n==0 else 255
                    ymin, ymax = psd.min(), psd.max()
                    period = 1 / freq[np.argmax(psd)]
                    x = self.normalize(freq, xmin, xmax) * (SIZEX*PIXEL - 10) + 5
                    y = (1-self.normalize(psd, ymin, ymax)) * (SIZEY*PIXEL - 10) + 5
                    draw.line(list(zip(x, y)), fill=self.get_color(c), width=1)
                    self.draw_title(draw, 3-n, "period from {stat} = {T:.2f}s".format(stat=name, T=period), color=c)
        del draw

    def draw_recurrence(self, e=0.1, steps=10):
        ''' https://stackoverflow.com/questions/33650371/recurrence-plot-in-python '''
        if self.analyzer.series == [] or len(self.analyzer.series[-1]) < 2:
            return

        size = min(SIZEX*PIXEL, SIZEY*PIXEL)
        segment = np.asarray(self.analyzer.series[-1])[-size:, self.analyzer.RECURRENCE_RANGE]
        vmin, vmax = segment.min(axis=0), segment.max(axis=0)
        # vmean = (vmax + vmin) / 2
        # d = vmax - vmin < 0.01
        # vmin[d], vmax[d] = vmean[d] - 0.01, vmean[d] + 0.01
        segment = self.normalize(segment, vmin, vmax)
        D = scipy.spatial.distance.squareform(np.log(scipy.spatial.distance.pdist(segment))) + np.eye(segment.shape[0]) * -100
        buffer = np.uint8(np.clip(-D/2, 0, 1) * 253)
        self.img = PIL.Image.frombuffer('L', buffer.shape, buffer, 'raw', 'L', 0, 1)

    def calc_fps(self):
        freq = 20 if self.samp_freq==1 else 200
        if self.automaton.gen == 0:
            self.last_time = time.time()
        elif self.automaton.gen % freq == 0:
            this_time = time.time()
            self.fps = freq / (this_time - self.last_time)
            self.last_time = this_time

    def change_b(self, i, d, s=12):
        b = self.world.params[self.show_kernel]['b'].copy()
        B = len(b)
        if B > 1 and i < B:
            b[i] = min(1, max(0, b[i] + Fraction(d, s)))
            #check if a least one fraction = 1
            # if b.count(1) >= 1:
            self.world.params[self.show_kernel]['b'] = b
            self.automaton.calc_kernel()
            self.check_auto_load()

    def adjust_b(self, d):
        B = len(self.world.params[self.show_kernel]['b'])
        if B <= 0:
            self.world.params[self.show_kernel]['b'] = [1]
        elif B >= 5:
            self.world.params[self.show_kernel]['b'] = self.world.params[self.show_kernel]['b'][0:5]
        elif KN == 1 and XN == 1:
            self.world.model['R'] = self.world.model['R'] * B // (B-d)
            temp_R = self.tx['R']
            self.tx['R'] = self.tx['R'] * (B-d) // B
            self.transform_world()
            self.world.model['R'] = temp_R
        self.automaton.calc_kernel()
        self.check_auto_load()

    def _recur_write_csv(self, dim, writer, cells):
        if dim < DIM-2:
            writer.writerow(["<{d}D>".format(d=DIM-dim)])
            for e in cells:
                self._recur_write_csv(dim+1, writer, e)
            writer.writerow(["</{d}D>".format(d=DIM-dim)])
        else:
            st = [ ['0' if c==0 else '{:.2f}'.format(c) for c in row] for row in cells]
            writer.writerows(st)
            #writer.writerow([])

    def copy_world(self, type='JSON'):
        if len(self.world_list) == 1:
            A = copy.deepcopy(self.world)
            A.crop()
            data_list = [ A.to_data() ]
        else:
            A = [ copy.deepcopy(world) for world in self.world_list ]
            data_list = [ A0.to_data() for A0 in A ]
        if type == 'JSON':
            to_save = data_list if len(data_list) > 1 else data_list[0]
            self.clipboard_st = json.dumps(to_save, separators=(',', ':'), ensure_ascii=False) + ','
        elif type == 'CSV':
            #if self.show_what in [0,4]: A = self.world.cells
            #elif self.show_what == 1: A = [self.get_kernel_array(self.automaton.potential)]
            #elif self.show_what == 2: A = [self.get_kernel_array(self.automaton.field)]
            #elif self.show_what == 3: A = [self.get_kernel_array(self.automaton.kernel)]
            #data = A.to_data()
            all_st = ""
            for data in data_list:
                data.pop('cells', None)
                for p in data['params']:
                    c0 = p.pop('c0', None)
                    c1 = p.pop('c1', None)
                    if c0: p['c0'], p['c1'] = c0, c1
                st = json.dumps(data, separators=(',', ':'), ensure_ascii=False) + ','
                st = st.replace('"params":[{', '"params":[\n{').replace('},{', '},\n{').replace('"', '').replace('{b:', '{b:[').replace(',m:', '],m:')
                stio = io.StringIO()
                writer = csv.writer(stio, delimiter=',', lineterminator='\n')
                for (i, ch) in enumerate(A.cells):
                    writer.writerow(["<Channel i={c}>".format(c=i)])
                    self._recur_write_csv(0, writer, np.round(ch, 2))
                    writer.writerow(["</Channel>"])
                all_st += '\n' + stio.getvalue()
            self.clipboard_st = all_st

        self.window.clipboard_clear()
        self.window.clipboard_append(self.clipboard_st)
        STATUS.append("> copied board to clipboard as "+type)

    def paste_world(self):
        try:
            st = self.clipboard_st = self.window.clipboard_get()
            if 'cells' in st:
                st = st.replace('\n', '').replace('\r', '').replace('\t', ' ').rstrip(', ')
                data = json.loads(st)
                if type(data) in [list]:
                    self.world_list = [ Board(list(reversed(SIZE))) for d in data ]
                    for world, d in zip(self.world_list, data):
                        self.load_part(world, Board.from_data(d))
                else:
                    self.load_part(self.world, Board.from_data(data))
                self.info_type = 'params'
                self.world_updated()
            elif '\t' in st or ',' in st:
                delim = '\t' if '\t' in st else ','
                stio = io.StringIO(st)
                #stio.setvalue(st)
                reader = csv.reader(stio, delimiter=delim, lineterminator='\n')
                cells = np.asarray([[float(c) if c != '' else 0 for c in row] for row in reader])
                self.load_part(self.world, Board.from_values(cells))
                self.world_updated()
            else:
                id = self.load_animal_code(self.world, st)
                if id is None:
                    id = self.load_found_animal_code(self.world, st)
                if id is not None:
                    self.info_type = 'animal'
                    self.world_updated()
        except (tk.TclError, ValueError, json.JSONDecodeError) as e:
            STATUS.append("> no valid JSON or CSV in clipboard")

    def save_world(self, is_seq=False):
        if len(self.world_list) == 1:
            A = copy.deepcopy(self.world)
            A.crop()
            data_list = [ A.to_data() ]
        else:
            A = [ copy.deepcopy(world) for world in self.world_list ]
            data_list = [ A0.to_data() for A0 in A ]
        try:
            if not os.path.exists(self.SAVE_ROOT):
                os.makedirs(self.SAVE_ROOT)
            if is_seq:
                self.file_seq += 1
            else:
                self.file_seq = 0
            path = os.path.join(self.SAVE_ROOT, str(self.file_seq))
            with open(path+'.rle', 'w', encoding='utf8') as file:
                file.write('#N '+A.long_name()+'\n')
                file.write('x = '+str(A.cells[0].shape[0])+', y = '+str(A.cells[0].shape[1])+', rule = Lenia('+A.params2st()+')\n')
                for data in data_list:
                    for rle in data['cells']:
                        file.write(rle.replace('$','$\n')+'\n')
            for data in data_list:
                data['cells'] = [[row if row.endswith('!') else row+'%' for row in rle.split('%')] for rle in data['cells']]
            with open(path+'.json', 'w', encoding='utf-8') as file:
                to_save = data_list if len(data_list) > 1 else data_list[0]
                json.dump(to_save, file, indent=4, ensure_ascii=False)
            with open(path+'.csv', 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow([self.analyzer.stats_fullname(x=x) for x in self.analyzer.STAT_HEADERS])
                writer.writerows([e for l in self.analyzer.series for e in l])
            STATUS.append("> data and image saved to '"+path+".*'")
            self.is_save_image = True
        except IOError as e:
            STATUS.append("I/O error({}): {}".format(e.errno, e.strerror))
    
    def shift_channel(self, d):
        new_shift = self.channel_shift + d
        self.channel_shift = new_shift % CN
        if self.channel_shift != new_shift:
            self.channel_group = (self.channel_group + d) % len(self.channelmaps)
        self.info_type = 'channel'

    def change_stat_axis(self, axis1, axis2, d):
        if self.stats_mode == 0:
            self.stats_mode = 1
        while True:
            axis1 = (axis1 + d) % len(self.analyzer.STAT_HEADERS)
            if axis1 != axis2 and axis1 > 2: break
        return axis1

    def toggle_auto_rotate_from_sampling(self):
        if self.auto_rotate_mode not in [3]:
            self.auto_rotate_mode = 3
            self.samp_gen = self.samp_freq if self.samp_freq > 1 else self.samp_gen
            self.samp_freq = 1
            self.is_auto_center = True
            self.info_type = 'angular'
        else:
            self.auto_rotate_mode = 0
            self.samp_freq = 1
            self.is_auto_center = False

    def reload_animal(self):
        # if self.last_load_animal:
        #     self.load_animal_id(self.world, self.animal_id)
        # else:
        #     self.load_found_animal_id(self.world, self.found_animal_id)
        self.load_part(self.world, self.fore)
        self.world_updated()
    
    def invert_world(self):
        self.automaton.is_inverted = not self.automaton.is_inverted
        self.world.model['T'] *= -1
        for k in KERNEL:
            self.world.params[k]['m'] = 1 - self.world.params[k]['m']
        for c in CHANNEL:
            self.world.cells[c] = 1 - self.world.cells[c]

    SHIFT_KEYS = {'asciitilde':'quoteleft', 'exclam':'1', 'at':'2', 'numbersign':'3', 'dollar':'4', 'percent':'5', 'asciicircum':'6', 'ampersand':'7', 'asterisk':'8', 'parenleft':'9', 'parenright':'0', 'underscore':'minus', 'plus':'equal', \
        'braceleft':'bracketleft', 'braceright':'bracketright', 'bar':'backslash', 'colon':'semicolon', 'quotedbl':'quoteright', 'less':'comma', 'greater':'period', 'question':'slash'}
    def key_press_event(self, event):
        ''' TKInter keys: https://www.tcl.tk/man/tcl8.6/TkCmd/keysyms.htm '''
        # Win: shift_l/r(0x1) caps_lock(0x2) control_l/r(0x4) alt_l/r(0x20000) win/app/alt_r/control_r(0x40000)
        # Mac: shift_l(0x1) caps_lock(0x2) control_l(0x4) meta_l(0x8,command) alt_l(0x10) super_l(0x40,fn)
        # print('keysym[{0.keysym}] char[{0.char}] keycode[{0.keycode}] state[{1}]'.format(event, hex(event.state))); return
        key = event.keysym
        state = event.state
        s = 's+' if state & 0x1 or (key.isalpha() and len(key)==1 and key.isupper()) else ''
        c = 'c+' if state & 0x4 or (not is_windows and state & 0x8) else ''
        a = 'a+' if state & 0x20000 else ''
        key = key.lower()
        if key in self.SHIFT_KEYS:
            key = self.SHIFT_KEYS[key]
            s = 's+'
        self.last_key = s + c + a + key
        self.is_internal_key = False

    def key_press_internal(self, key):
        self.last_key = key
        self.is_internal_key = True

    if DIM == 2:
        ANIMAL_KEY_LIST = {'1':'O2u', '2':'OG2g', '3':'OV2u', '4':'P4ap', '5':'PS3ap', '6':'2S2:2', '7':'1P4oal', '8':'2PG1v', '9':'3H3t', '0':'~gldr', \
            's+1':'3GH2n', 's+2':'3GG2g', 's+3':'K5s', 's+4':'K7a', 's+5':'K9a', 's+6':'3R5i', 's+7':'3R6s:4', 's+8':'2D10rb', 's+9':'4F14x9xR5r', 's+0':'~ggun', \
            'c+1':'4Q5m:2', 'c+2':'3ECv', 'c+3':'K4s', 'c+4':'K4v', 'c+5':'2L4m:2', 'c+6':'3R6n', 'c+7':'4D10v:2', 'c+8':'', 'c+9':'', 'c+0':'bbug',
            'a+1':'2F5l', 'a+2':'3L6v', 'a+3':'2L3i', 'a+4':'3F8t', 'a+5':'4R5n', 'a+6':'3R5r:2', 'a+7':'3R5t:2', 'a+8':'2D6s', 'a+9':'3R6i', 'a+0':'4D10v:2',
            's+a+1':'P4cl', 's+a+2':'P4cs', 's+a+3':'P4cp', 's+a+4':'P4sp', 's+a+5':'P4cf', 's+a+6':'2PG1v', 's+a+7':'OG2g', 's+a+8':'OV2u', 's+a+9':'OG2r', 's+a+0':'3GG2g'}
    elif DIM == 3:
        ANIMAL_KEY_LIST = {'1':'4Gu2s', '2':'2Pl8l', '3':'2Pl4t', '4':'4As1a', '5':'3As1l', '6':'3As2l', '7':'2As2v', '8':'2MePs', '9':'', '0':''}
    elif DIM == 4:
        ANIMAL_KEY_LIST = {'1':'3Hy2v', '2':'', '3':'', '4':'', '5':'', '6':'', '7':'', '8':'', '9':'', '0':''}
    else:
        ANIMAL_KEY_LIST = {}

    def process_key(self, k):
        global STATUS
        is_life = (self.world.model.get('P') == 1)
        inc_or_dec     = 1 if 's+' not in k else -1
        inc_10_or_1    = 10 if 's+' not in k else 1
        inc_10_or_100  = 10 if 's+' not in k else 100
        inc_5_or_1     = 5  if 's+' not in k else 1
        rot_15_or_1    = 90 if is_life else 15 if 's+' not in k else 1
        inc_1_or_5     = 1 if 's+' not in k else 5
        inc_1_or_10    = 1 if 's+' not in k else 10
        inc_mul_or_not = 1 if 's+' not in k else 0
        double_or_not  = 2 if 's+' not in k else 1
        inc_or_not     = 0 if 's+' not in k else 1

        #inc_10_or_1   = (10 if 's+' not in k else 1) if 'c+' not in k else 0 
        #inc_big_or_not = 0 if 'c+' not in k else 1

        is_ignore = False
        if not self.is_internal_key and k not in ['backspace', 'delete']:
            self.finish_search()

        #run [esc enter space c+g]
        if k in ['escape']: self.is_closing = True; self.close()
        elif k in ['enter', 'return']: self.is_run = not self.is_run; self.run_counter = -1; self.info_type = 'time'
        elif k in [' ', 'space']: self.is_run = True; self.run_counter = 1; self.info_type = 'time'
        elif k in ['c+space']: self.is_run = True; self.run_counter = self.samp_freq; self.info_type = 'time'
        elif k in ['s+c+g']:
            if self.automaton.has_gpu:
                self.automaton.is_gpu = not self.automaton.is_gpu
        elif k in ['c+tab']: self.is_advanced_menu = not self.is_advanced_menu; self.create_menu()
        #sampling,symmetry [[ ] \]
        elif k in ['bracketright', 's+bracketright']: self.samp_freq = self.samp_freq + (4 if self.samp_freq==1 and inc_5_or_1==5 else inc_5_or_1); self.info_type = 'time'
        elif k in ['bracketleft',  's+bracketleft']:  self.samp_freq = self.samp_freq - inc_5_or_1; self.info_type = 'time'
        elif k in ['a+bracketright']: self.samp_freq = int(round((round(self.samp_freq / self.analyzer.period_gen) + 1) * self.analyzer.period_gen)); self.info_type = 'time'
        elif k in ['a+bracketleft']: self.samp_freq = max(1, int(round((round(self.samp_freq / self.analyzer.period_gen) - 1) * self.analyzer.period_gen))); self.info_type = 'time'
        elif k in ['c+bracketright']: self.samp_sides += 1; self.info_type = 'angular'
        elif k in ['c+bracketleft']: self.samp_sides -= 1; self.info_type = 'angular'
        elif k in ['backslash']: self.toggle_auto_rotate_from_sampling()
        elif k in ['s+backslash']: self.samp_freq = 1; self.info_type = 'time'
        elif k in ['c+backslash']: self.is_samp_clockwise = not self.is_samp_clockwise; self.info_type = 'angular'
        #display [< > tab `]
        elif k in ['s+period'] and not self.is_show_rgb(): self.colormap_id = (self.colormap_id + 1) % len(self.colormaps)
        elif k in ['s+comma']  and not self.is_show_rgb(): self.colormap_id = (self.colormap_id - 1) % len(self.colormaps)
        elif k in ['s+period'] and self.is_show_rgb(): self.shift_channel(+1)
        elif k in ['s+comma']  and self.is_show_rgb(): self.shift_channel(-1)
        elif k in ['tab', 's+tab']: self.show_what = (self.show_what + inc_or_dec) % 5
        elif k in ['quoteleft', 's+quoteleft']: self.set_show(inc_or_dec); self.info_type = 'params'
        elif k in ['s+c+tab']: self.show_what = 0; self.show_group = 0; self.show_kernel = 0; self.colormap_id = 0; self.channel_group = 0; self.channel_shift = 0; self.info_type = 'channel'
        # elif k in ['c+tab', 's+c+tab']: self.show_what = (self.show_what + inc_or_dec) % 8
        #params [qa ws]
        elif k in ['q', 's+q']: self.world.params[self.show_kernel]['m'] += inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['a', 's+a']: self.world.params[self.show_kernel]['m'] -= inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['w', 's+w']: self.world.params[self.show_kernel]['s'] += inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['s', 's+s']: self.world.params[self.show_kernel]['s'] -= inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['c+t']: self.world.params[self.show_kernel]['h'] = min(1.0, self.world.params[self.show_kernel]['h'] + 0.1); self.analyzer.new_segment(); self.info_type = 'params'
        elif k in ['c+g']: self.world.params[self.show_kernel]['h'] = max(0.1, self.world.params[self.show_kernel]['h'] - 0.1); self.analyzer.new_segment(); self.info_type = 'params'
        elif k in ['c+r']: self.world.params[self.show_kernel]['r'] = min(1.0, self.world.params[self.show_kernel]['r'] + 0.1); self.analyzer.new_segment(); self.info_type = 'params'
        elif k in ['c+f']: self.world.params[self.show_kernel]['r'] = max(0.5, self.world.params[self.show_kernel]['r'] - 0.1); self.analyzer.new_segment(); self.info_type = 'params'
        #model [ed rf tg]
        elif k in ['e', 's+e']: self.world.model['P'] = max(0, self.world.model['P'] + inc_10_or_1); self.info_type = 'info'
        elif k in ['d', 's+d']: self.world.model['P'] = max(0, self.world.model['P'] - inc_10_or_1); self.info_type = 'info'
        elif k in ['r', 's+r']: self.tx['R'] = max(1, self.tx['R'] + inc_5_or_1); self.transform_world(); self.info_type = 'info'
        elif k in ['f', 's+f']: self.tx['R'] = max(1, self.tx['R'] - inc_5_or_1); self.transform_world(); self.info_type = 'info'
        elif k in ['t', 's+t']: self.world.model['T'] = max(1, self.world.model['T'] *  double_or_not + inc_or_not); self.analyzer.new_segment(); self.info_type = 'info'
        elif k in ['g', 's+g']: self.world.model['T'] = max(1, self.world.model['T'] // double_or_not - inc_or_not); self.analyzer.new_segment(); self.info_type = 'info'
        elif k in ['s+c+d']: self.world.model['P'] = 0; self.info_type = 'info'
        elif k in ['s+c+r']: self.tx['R'] = DEF_R; self.transform_world(); self.info_type = 'info'
        elif k in ['s+c+f']: self.tx['R'] = self.fore.model['R'] if self.fore else DEF_R; self.transform_world(); self.info_type = 'info'
        elif k in ['s+c+t']: self.world.model['T'] = self.world.model['T'] / 2; self.analyzer.new_segment(); self.info_type = 'info'
        #param B [yuiop ;]
        elif k in ['y', 's+y']: self.change_b(0, inc_or_dec, 12); self.info_type = 'params'
        elif k in ['u', 's+u']: self.change_b(1, inc_or_dec, 12); self.info_type = 'params'
        elif k in ['i', 's+i']: self.change_b(2, inc_or_dec, 12); self.info_type = 'params'
        elif k in ['o', 's+o']: self.change_b(3, inc_or_dec, 12); self.info_type = 'params'
        elif k in ['p', 's+p']: self.change_b(4, inc_or_dec, 12); self.info_type = 'params'
        elif k in ['a+y', 's+a+y']: self.change_b(0, inc_or_dec, 24); self.info_type = 'params'
        elif k in ['a+u', 's+a+u']: self.change_b(1, inc_or_dec, 24); self.info_type = 'params'
        elif k in ['a+i', 's+a+i']: self.change_b(2, inc_or_dec, 24); self.info_type = 'params'
        elif k in ['a+o', 's+a+o']: self.change_b(3, inc_or_dec, 24); self.info_type = 'params'
        elif k in ['a+p', 's+a+p']: self.change_b(4, inc_or_dec, 24); self.info_type = 'params'
        elif k in ['semicolon']:   self.world.params[self.show_kernel]['rings'].append(EMPTY_RING.copy()); self.info_type = 'params'
        elif k in ['s+semicolon']: self.world.params[self.show_kernel]['rings'].pop(); self.info_type = 'params'
        #search [c+qa]
        elif k in ['c+q', 's+c+q']: self.is_search_small = 's+' in k; self.toggle_search(+1)
        elif k in ['c+a', 's+c+a']: self.is_search_small = 's+' in k; self.toggle_search(-1)
        #options [c+yuiop]
        elif k in ['c+y', 's+c+y']: self.world.model['kn'] = (self.world.model['kn'] + inc_or_dec - 1) % len(self.automaton.kernel_core) + 1; self.automaton.calc_kernel(); self.info_type = 'kn'
        elif k in ['c+u', 's+c+u']: self.world.model['gn'] = (self.world.model['gn'] + inc_or_dec - 1) % len(self.automaton.growth_func) + 1; self.info_type = 'gn'
        elif k in ['c+i']: self.automaton.soft_clip_level = (self.automaton.soft_clip_level + 1) % 10; self.world.model['vmin'] = EPSILON if self.automaton.soft_clip_level == 0 else ALIVE_THRESHOLD
        elif k in ['c+o']: self.invert_world()
        elif k in ['c+p']: self.automaton.is_arita_mode = not self.automaton.is_arita_mode
        elif k in ['s+c+i']: self.automaton.mask_rate = (self.automaton.mask_rate + 1) % 10
        elif k in ['s+c+o']: self.automaton.add_noise = (self.automaton.add_noise + 1) % 11
        elif k in ['s+c+p']: self.automaton.mask_rate = 0; self.automaton.add_noise = 0
        #shift [c+arrow/pg]
        elif k in ['left',  's+left' ]: self.tx['shift'][X_AXIS] -= inc_10_or_1; self.transform_world()
        elif k in ['right', 's+right']: self.tx['shift'][X_AXIS] += inc_10_or_1; self.transform_world()
        elif k in ['down',  's+down' ]: self.tx['shift'][Y_AXIS] += inc_10_or_1; self.transform_world()
        elif k in ['up',    's+up',  ]: self.tx['shift'][Y_AXIS] -= inc_10_or_1; self.transform_world()
        elif k in ['pagedown', 's+pagedown', 'next',  's+next' ] and DIM>2: self.tx['shift'][self.z_axis] -= inc_10_or_1; self.transform_world()
        elif k in ['pageup',   's+pageup',   'prior', 's+prior'] and DIM>2: self.tx['shift'][self.z_axis] += inc_10_or_1; self.transform_world()
        #rotate [c+arrow/pg]
        elif k in ['c+left',  's+c+left' ]: self.tx['rotate'][2] -= rot_15_or_1; self.transform_world()
        elif k in ['c+right', 's+c+right']: self.tx['rotate'][2] += rot_15_or_1; self.transform_world()
        elif k in ['c+down',  's+c+down' ]: self.tx['rotate'][1] += rot_15_or_1; self.transform_world()
        elif k in ['c+up',    's+c+up'   ]: self.tx['rotate'][1] -= rot_15_or_1; self.transform_world()
        elif k in ['c+pagedown', 's+c+pagedown', 'c+next',  's+c+next' ] and DIM>2: self.tx['rotate'][0] -= rot_15_or_1; self.transform_world()
        elif k in ['c+pageup',   's+c+pageup',   'c+prior', 's+c+prior'] and DIM>2: self.tx['rotate'][0] += rot_15_or_1; self.transform_world()
        #slice (DIM>2) [home/end]
        elif k in ['home', 's+home'] and DIM>2: self.is_show_slice = True; self.z_slices[self.z_axis] = (self.z_slices[self.z_axis] + inc_10_or_1) % SIZE[self.z_axis]; self.info_type = 'slice'
        elif k in ['end',  's+end' ] and DIM>2: self.is_show_slice = True; self.z_slices[self.z_axis] = (self.z_slices[self.z_axis] - inc_10_or_1) % SIZE[self.z_axis]; self.info_type = 'slice'
        elif k in ['c+home'] and DIM>2: self.is_show_slice = True; self.z_slices = [MID[DIM-1-d] for d in range(DIM-2)]; self.z_axis = 0; self.info_type = 'slice'
        elif k in ['c+end' ] and DIM>2: self.is_show_slice = not self.is_show_slice
        #dimension (DIM>3) [s+c+home/end]
        elif k in ['s+c+home'] and DIM>2: self.z_axis = (self.z_axis + 1) % (DIM-2) if DIM>3 else 0; self.info_type = 'slice'
        elif k in ['s+c+end' ] and DIM>2: self.z_axis = (self.z_axis - 1) % (DIM-2) if DIM>3 else 0; self.info_type = 'slice'
        #flip [=]
        elif k in ['equal']:     self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['s+equal']:   self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
        elif k in ['c+equal']:   self.tx['flip'] = 2 if self.tx['flip'] != 2 else -1; self.transform_world()
        elif k in ['s+c+equal']: self.tx['flip'] = 3 if self.tx['flip'] != 3 else -1; self.transform_world()
        elif k in ['minus']:     self.tx['flip'] = 4 if self.tx['flip'] != 4 else -1; self.transform_world()
        elif k in ['s+minus']:   self.tx['flip'] = 5 if self.tx['flip'] != 5 else -1; self.transform_world()
        elif k in ['c+minus']:   self.tx['flip'] = 6 if self.tx['flip'] != 6 else -1; self.transform_world()
        #autocenter [']
        elif k in ['quoteright']: self.is_auto_center = not self.is_auto_center
        elif k in ['s+quoteright']: self.auto_rotate_mode = (self.auto_rotate_mode + 1) % 3 if DIM==2 else 0
        elif k in ['c+quoteright', 's+c+quoteright']: self.polar_mode = (self.polar_mode + inc_or_dec) % 5 if DIM==2 else 0
        # elif k in ['c+quoteright']: self.is_auto_center = True; self.auto_rotate_mode = 2; self.analyzer.is_calc_symmetry = True; self.stats_mode = 5; self.samp_freq = int(round(self.analyzer.period_gen*2))
        # elif k in ['s+c+quoteright']: self.is_auto_center = False; self.auto_rotate_mode = 0; self.analyzer.is_calc_symmetry = False; self.stats_mode = 0; self.samp_freq = 1
        #animals [zxcv 1-0]
        elif k in ['z']: self.reload_animal(); self.info_type = 'animal'
        elif k in ['c']: 
            if CN==1 and KN==1:
                self.load_animal_id(self.world, self.animal_id - inc_1_or_10)
            else:
                self.load_found_animal_id(self.world, self.found_animal_id - inc_1_or_10)
            self.world_updated()
            self.info_type = 'animal'
        elif k in ['v']: 
            if CN==1 and KN==1:
                self.load_animal_id(self.world, self.animal_id + inc_1_or_10)
            else:
                self.load_found_animal_id(self.world, self.found_animal_id + inc_1_or_10)
            self.world_updated()
            self.info_type = 'animal'
        elif k in ['s+c']:
            if CN==1 and KN==1:
                self.search_animal(self.world, 'family: ', -1)
            else:
                self.search_found_animal(self.world, '*', -1)
            self.info_type = 'animal'
        elif k in ['s+v']:
            if CN==1 and KN==1:
                self.search_animal(self.world, 'family: ', +1)
            else:
                self.search_found_animal(self.world, '*', +1)
            self.info_type = 'animal'
        elif k in ['s+z']: self.found_animal_id = len(self.found_animal_data) if self.found_animal_id==0 else 0; self.load_found_animal_id(self.world, self.found_animal_id); self.world_updated(); self.info_type = 'animal'
        elif k in ['c+backspace', 'c+delete']: self.delete_found_animal(code=self.world.names['code']); self.load_found_animal_id(self.world, self.found_animal_id); self.world_updated(); self.info_type = 'animal'
        elif k in ['x', 's+x']: self.load_part(self.world, self.fore, is_random=True, is_replace=False, repeat=inc_1_or_5); self.world_updated(is_random=True)
        elif k in ['c+z']: self.is_auto_load = not self.is_auto_load
        elif k in ['s+c+z']: self.read_animals(); self.read_found_animals(); self.create_menu()
        elif k in ['s+c+x']: self.is_layer_mode = not self.is_layer_mode
        elif k in [m+str(i) for i in range(10) for m in ['','s+','c+','s+c+','a+','s+a+']]: self.load_animal_code(self.world, self.ANIMAL_KEY_LIST.get(k)); self.world_updated(); self.info_type = 'animal'
        #multi-species [b]
        elif k in ['b', 's+b']: i = self.world_list.index(self.world); self.world = self.world_list[(i + inc_or_dec) % len(self.world_list)]
        elif k in ['c+b']: self.world = Board(list(reversed(SIZE))); self.world_list.append(self.world); self.automaton = Automaton(self.world); self.automaton_list.append(self.automaton)
        elif k in ['s+c+b'] and len(self.world_list) > 1: self.world = self.world_list.pop()
        #random [del n m]
        elif k in ['backspace', 'delete']: self.clear_world()
        elif k in ['n', 's+n', 'c+n', 's+c+n']: self.random_world(is_reseed=(k=='s+n'), density_mode=2 if k=='c+n' else 0 if k=='s+c+n' else 1)
        # elif k in ['n', 's+n', 'c+n', 's+c+n']:
        #     #self.random_density += (1 if 'c+' in k else 5)*inc_or_dec
        #     d = []
        #     for i in range(10):
        #         #self.random_world()
        #         self.random_world(is_reseed='s+' in k)
        #         d.append(sum(self.world.cells).sum()/np.prod(SIZE))
        #     print("{}, {:.2f}".format(self.random_density, sum(d)/len(d)))
        elif k in ['m']: self.random_params(); self.random_world(); self.info_type = 'params'
        elif k in ['s+m']: self.reload_animal(); self.random_params(is_incremental=True); self.info_type = 'params'
        elif k in ['c+m']: self.toggle_search(0); self.is_run = True; self.info_type = 'params'
        elif k in ['s+c+m']: self.search_algo = (self.search_algo + 1) % 7; self.info_type = 'search'
        #copy,paste,save,record [c+xcv c+s c+w]
        elif k in ['c+c']: self.copy_world(type='JSON')
        elif k in ['c+x']: self.copy_world(type='CSV')
        elif k in ['c+v']: self.paste_world()
        elif k in ['c+s', 's+c+s']: self.save_world(is_seq='s+' in k)
        elif k in ['c+w', 's+c+w']: self.is_run = self.recorder.toggle_recording(is_save_frames='s+' in k)
        #stats [hjkl]
        elif k in ['h', 's+h']: self.markers_mode = (self.markers_mode + inc_or_dec) % 8
        elif k in ['c+h']: self.is_show_fps = not self.is_show_fps
        elif k in ['j', 's+j']: self.stats_mode = (self.stats_mode + inc_or_dec) % 7; self.info_type = 'stats'
        elif k in ['k', 's+k']: self.stats_x = self.change_stat_axis(self.stats_x, self.stats_y, inc_or_dec); self.info_type = 'stats'
        elif k in ['l', 's+l']: self.stats_y = self.change_stat_axis(self.stats_y, self.stats_x, inc_or_dec); self.info_type = 'stats'
        elif k in ['c+j']: self.analyzer.clear_segment()
        elif k in ['a+j']: self.stats_mode = 5  # periodogram
        elif k in ['s+c+j']: self.analyzer.clear_series()
        elif k in ['c+k']: self.analyzer.trim_segment = (self.analyzer.trim_segment + inc_or_dec) % 3
        elif k in ['c+l']: self.is_group_params = not self.is_group_params
        elif k in ['s+c+k']: self.stats_mode = 1; self.stats_x_name = 'm'; self.stats_y_name = 'g'; self.analyzer.trim_segment = 1; self.info_type = 'stats'
        elif k in ['s+c+l']: self.stats_mode = 2; self.stats_x_name = 'x'; self.stats_y_name = 'y'; self.analyzer.trim_segment = 2; self.info_type = 'stats'
        #info [,./]
        elif k in ['comma']: self.info_type = 'animal'
        elif k in ['period']: self.info_type = 'params'
        elif k in ['slash']: self.info_type = 'info'
        elif k in ['s+slash']: self.info_type = 'angular'
        # elif k in ['c+period']: self.info_type = 'kernel'
        # elif k in ['c+slash']: self.info_type = 'size'
        #misc
        elif k in ['c+period', 's+c+period']: self.analyzer.object_distance = np.clip(round(self.analyzer.object_distance - inc_5_or_1/100, 2), 0, 1); self.analyzer.detect_objects(); self.info_type = 'object'
        elif k in ['c+slash',  's+c+slash']:  self.analyzer.object_distance = np.clip(round(self.analyzer.object_distance + inc_5_or_1/100, 2), 0, 1); self.analyzer.detect_objects(); self.info_type = 'object'
        # elif k in ['c+comma']: pass
        # elif k in ['c+slash']: m = self.menu.children[self.menu_values['animal'][0]].children['!menu']; m.post(self.window.winfo_rootx(), self.window.winfo_rooty())
        elif k.endswith('_l') or k.endswith('_r'): is_ignore = True
        else: self.excess_key = k

        #print(self.world_list.index(self.world), len(self.world_list), self.automaton_list.index(self.automaton), len(self.automaton_list))
        
        if self.polar_mode not in [0] or self.auto_rotate_mode in [2]:
            self.analyzer.is_calc_symmetry = True
        else:
            self.analyzer.is_calc_symmetry = False
        if self.stats_mode in [5]:
            self.analyzer.is_calc_psd = True
        if self.auto_rotate_mode not in [0]:
            self.is_auto_center = True

        self.samp_freq = max(1, self.samp_freq)
        self.samp_sides = max(1, self.samp_sides)
        self.samp_rotate = (-1 if self.is_samp_clockwise else +1) * 360/self.samp_sides/self.samp_gen*self.world.model['T'] if self.auto_rotate_mode in [3] else 0
        if not is_ignore and self.is_loop:
            for k in KERNEL:
                self.roundup(self.world.params[k])
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)
            self.update_menu()

        # if not is_ignore:
            # v1 = self.target.cells.reshape(-1, 1)
            # v2 = self.world.cells.reshape(-1, 1)
            # print('val: {:.3f}'.format(-np.linalg.norm(v1 - v2)))

    def set_show(self, inc_or_dec):
        #0 1 2 3+0 3+1... 3+k 0
        n = self.show_kernel + self.show_group
        n = (n + inc_or_dec) % (len(KERNEL) + 2)
        self.show_kernel = max(n-2, 0)
        self.show_group = min(n, 2)

    def combine_worlds(self, world_sum, worlds_list):
        ex = np.clip(worlds_list[0].cells + worlds_list[1].cells - 1, 0, 1) / 2
        worlds_list[0].cells -= ex
        worlds_list[1].cells -= ex
        world_sum.cells = worlds_list[0].cells + worlds_list[1].cells

    def roundup(self, A):
        for (k,x) in A.items():
            if type(x)==float:
                A[k] = round(x, ROUND)

    def get_acc_func(self, key, acc, animal_id=None):
        acc = acc if acc else key if key else None
        ctrl = 'Ctrl+' if (is_windows or acc in ['c+Space','c+Q','c+H','c+Tab']) else 'Command+'
        if acc: acc = acc.replace('s+','Shift+').replace('c+',ctrl).replace('m+','Cmd+').replace('a+','Slt+')
        if animal_id:
            func = lambda:self.load_animal_id(self.world, int(animal_id))
        else:
            func = lambda:self.key_press_internal(key.lower()) if key else None
        state = 'normal' if key or animal_id else tk.DISABLED
        return {'accelerator':acc, 'command':func, 'state':state}
    def create_submenu(self, parent, items):
        m = tk.Menu(parent, tearoff=True)
        m.seq = 0
        is_last_sep = True
        for i in items:
            if i is None or i=='':
                if not is_last_sep:
                    m.add_separator()
                    m.seq += 1
                    is_last_sep = True
            elif type(i) in [tuple, list]:
                m.add_cascade(label=i[0], menu=self.create_submenu(m, i[1]))
                m.seq += 1
                is_last_sep = False
            else:
                first, text, key, acc, *_ = i.split('|') + ['']*2
                if acc=='bar': acc = '|'
                kind, name = first[:1], first[1:]
                if self.is_advanced_menu or (not self.is_advanced_menu and not text.startswith('*')):
                    is_last_sep = False
                    text = text.lstrip('*')
                    if (text.startswith('#') and DIM==2) or (text.startswith('$') and DIM>2):
                        key = None
                    text = text.lstrip('$#')
                    m.seq += 1
                    if first=='':
                        m.add_command(label=text, **self.get_acc_func(key, acc))
                    elif kind=='^':
                        self.menu_vars[name] = tk.BooleanVar(value=self.get_nested_attr(name))
                        m.add_checkbutton(label=text, variable=self.menu_vars[name], **self.get_acc_func(key, acc))
                    elif kind=='@':
                        self.menu_values[name] = (m._name, m.seq, text)
                        m.add_command(label='', **self.get_acc_func(key, acc)) # background='dark green', foreground='white'
                    elif kind=='#':
                        self.menu_params[name] = (m._name, m.seq, text)
                        m.add_command(label='', **self.get_acc_func(key, acc)) # state=tk.DISABLED, background='navy', foreground='white')
                    elif kind=='&':
                        m.add_command(label=text, **self.get_acc_func(key, acc, animal_id=name))
        if is_last_sep:
            m.delete(m.seq)
        return m
    def get_animal_nested_list(self):
        if self.animal_data is None or self.animal_data == []:
            return []
        root = []
        stack = [root]
        id = 0
        for data in self.animal_data:
            code = data['code']
            if code.startswith('>'):
                next_level = int(code[1:]) - 2  #start from family or lower
                if next_level >= 1:
                    d = len(stack) - next_level
                    for i in range(d):
                        stack.pop()
                    for i in range(max(-d, 0) + 1):
                        new_list = ("{name} {cname}".format(**data), [])
                        stack[-1].append(new_list)
                        stack.append(new_list[1])
            else:
                stack[-1].append("&{id}|{code} - {name} {cname}|".format(id=id, **data))
            id += 1
        return root

    def get_nested_attr(self, name):
        obj = self
        for n in name.split('.'):
            obj = getattr(obj, n)
        return obj
    def get_value_text(self, name):
        show_what_names = ["World","Potential","Field","Kernel","Objects"]
        if name=='animal': return '#'+str(self.animal_id+1)+' '+self.world.long_name()
        elif name=='kn': return ["Polynomial","Exponential","Step","Leaky Exponential"][self.world.model.get('kn') - 1]
        elif name=='gn': return ["Polynomial","Exponential","Step"][self.world.model.get('gn') - 1]
        elif name=='colormap_id': return ["Turbo","Turbo(Green)","Paul Tol's Rainbow","Jet","Life(Purple)","Life(Green)","White/black","Black/white"][self.colormap_id]    # "Pale blue/red","Pale green/purple","Pale yellow/green",
        # elif name=='show_what': return ["World","Potential","Field","Kernel","World FFT","Potential FFT","Gradient","Change"][self.show_what]
        elif name=='show_what': return show_what_names[self.show_what]
        elif name=='polar_mode': return ["Off","Symmetry","Polar","History","Strength"][self.polar_mode]
        elif name=='auto_rotate_mode': return ["Off","Arrow","Symmetry","Sampling"][self.auto_rotate_mode]
        elif name=='markers_mode': 
            st = []
            if self.markers_mode in [0,1,2,3]: st.append("Grid")
            if self.markers_mode in [0,1,4,5]: st.append("Legend")
            if self.markers_mode in [1,3,5,7]: st.append("Marks")
            return ",".join(st) if st != [] else "None"
        elif name=='stats_mode': return ["None","Corner","Overlay","Segment","All segments","Periodogram","Recurrence plot"][self.stats_mode]
        elif name=='stats_x': return self.analyzer.stats_fullname(i=self.stats_x)
        elif name=='stats_y': return self.analyzer.stats_fullname(i=self.stats_y)
        elif name=='z_axis': return str(DIM-self.z_axis)
        elif name=='mask_rate': return "{rate}%".format(rate=self.automaton.mask_rate * 10)
        elif name=='add_noise': return "{rate}%".format(rate=self.automaton.add_noise * 10)
        elif name=='search_algo': return ["Global search","Depth search","Breadth search","Depth+breadth search","Genetic algo on avg({stat})","Genetic algo on stdev({stat})","Genetic algo on max({stat})"][self.search_algo].format(stat=self.stats_x_name)
        elif name=='show_kernel': return self.show_kernel
        elif name=='show_group': return ["Average","Channel","Kernel #"+str(self.show_kernel)][self.show_group]  #"Sum","Average",
        elif name=='trim_segment': return ["Unlimited","Short","Long"][self.analyzer.trim_segment]
        elif name=='soft_clip': return self.SOFT_CLIP_NAME_LIST[self.automaton.soft_clip_level]
    def update_menu(self):
        for name in self.menu_vars:
            self.menu_vars[name].set(self.get_nested_attr(name))
        for (name, info) in self.menu_params.items():
            # value = '['+Board.fracs2st(self.world.params[name])+']' if name=='b' else self.world.params[name]
            # self.menu.children[info[0]].entryconfig(info[1], label='{text} ({param} = {value})'.format(text=info[2], param=name, value=value))
            value = self.get_nested_attr(name)
            self.menu.children[info[0]].entryconfig(info[1], label="{text} [{value}]".format(text=info[2], value=value))
        for (name, info) in self.menu_values.items():
            value = self.get_value_text(name)
            self.menu.children[info[0]].entryconfig(info[1], label="{text} [{value}]".format(text=info[2], value=value))

    PARAM_TEXT = {'m':'Field center', 's':'Field width', 'R':'Space units', 'T':'Time units', 'dr':'Space step', 'dt':'Time step', 'b':'Kernel peaks'}
    VALUE_TEXT = {'animal':'Lifeform', 'kn':'Kernel core', 'gn':'Field func', 'show_what':'Show', 'colormap_id':'Colors'}
    def create_menu(self):
        ''' menu item format: variable|text|key(actual)|key(displayed)
        variable: ^=checkbox, @=show value, #=show param, &=animal item
        text: *=advanced menu only, $=2D only(else disabled), #=3D+ only(else disabled)
        key(actual): item disabled if omitted
        key(displayed): optional '''

        self.menu_vars = {}
        self.menu_params = {}
        self.menu_values = {}
        self.menu = tk.Menu(self.window, tearoff=True)
        self.window.config(menu=self.menu)

        self.menu.add_cascade(label='Lenia', menu=self.create_submenu(self.menu, [
            '^is_run|Running|Return', '|Once|Space'] + 
            (['^automaton.is_gpu|Use GPU|s+c+G', '|*(GPU: '+self.automaton.gpu_thr._device.name+')|'] if self.automaton.has_gpu else ['|No GPU available|']) + [None,
            '@show_what|Display|Tab', '@show_kernel|*Kernel|QuoteLeft|`', '@colormap_id|Colors|s+Period|>', None,
            '|*Show lifeform name|Comma|,', '|*Show params|Period|.', '|*Show info|Slash|/', '|*Show auto-rotate info|s+Slash|?', None,
            '|Save data & image|c+S', '|*Save next in sequence|s+c+S', 
            '^recorder.is_recording|Record video & gif|c+W', '|*Record with frames saved|s+c+W', None,
            '^is_advanced_menu|Advanced menu|c+Tab', None,
            '|Quit|Escape']))

        self.menu.add_cascade(label='Edit', menu=self.create_submenu(self.menu, [
            '|Clear|Backspace', '|Random|N', '|*Random (last seed)|s+N', '|Random cells & params|M', '|*Random cells & params (incremental)|s+M', None,
            '|Flip vertically|Equal|=', '|Flip horizontally|s+Equal|+',
            '|Mirror horizontally|c+Equal|c+=', '|Mirror flip|s+c+Equal|c++', '|Erase half|Minus|-', None,
            '|Copy|c+C', '|*Copy as CSV|c+X', '|Paste|c+V', None,
            '^is_auto_load|*Auto put mode|c+Z', '^is_layer_mode|*Layer mode|s+c+X']))

        # for (key, code) in self.ANIMAL_KEY_LIST.items():
            # id = self.get_animal_id(code)
            # if id: items2.append('|{name} {cname}|{key}'.format(**self.animal_data[id], key=key))
        self.menu.add_cascade(label='Lifeform', menu=self.create_submenu(self.menu, [
            '|Place at center|Z', '|Add at random|X',
            '|Previous|C', '|Next|V', '|Previous family|s+C', '|Next family|s+V', None,
            '|*Previous found|s+B', '|*Next found|B', '|*First/last found|c+B', '|*Delete this found|c+Backspace', None,
            '|*Start auto search (any key to stop)|c+M', '@search_algo|*Search algorithm|s+c+M', None,
            '|Shortcuts 1-10|1', '|Shortcuts 11-20|s+1', '|Shortcuts 21-30|c+1', None,
            '|*Reload list|s+c+Z']))

        self.menu.add_cascade(label='List', menu=self.create_submenu(self.menu, 
            self.get_animal_nested_list()))

        self.menu.add_cascade(label='Space', menu=self.create_submenu(self.menu, [
            '^is_auto_center|Auto-center mode|QuoteRight|\'', None,
            '|(Small adjust)||s+Up', #'|(Large adjust)||m+Up',
            '|Move up|Up', '|Move down|Down', '|Move left|Left', '|Move right|Right',
            '|#Move front|PageUp', '|#Move back|PageDown', None,
            '|*#(Small adjust)||s+Home', '|*#Slice front|Home', '|*#Slice back|End', '|*#Center slice|c+Home', 
            '^is_show_slice|*#Show Z slice|c+End', '@z_axis|*#Change Z axis|s+c+Home']))

        self.menu.add_cascade(label='Polar', menu=self.create_submenu(self.menu, [
            '@polar_mode|$Polar mode|c+QuoteRight|c+\'', '@auto_rotate_mode|*$Auto-rotate by|s+QuoteRight|"', None] +
            (['|(Small adjust)||s+c+Up', '|Rotate anti-clockwise|c+Up', '|Rotate clockwise|c+Down', None] 
            if DIM == 2 else
            ['|(Small adjust)||s+c+Up', '|Rotate right|c+Right', '|Rotate left|c+Left', 
            '|Rotate up|c+Up', '|Rotate down|c+Down',
            '|Rotate anti-clockwise|c+PageUp', '|Rotate clockwise|c+PageDown', None])
            + ['|*(Small adjust)||s+]', '|*Sampling period + 10|BracketRight|]', '|*Sampling period - 10|BracketLeft|[', 
            '|*Clear sampling|s+BackSlash|bar', '|*Run one sampling period|c+Space', None,
            '|*$Auto-rotate by sampling|BackSlash|\\', '|*$Symmetry axes + 1|c+BracketRight|c+]', 
            '|*$Symmetry axes - 1|c+BracketLeft|c+[', '^is_samp_clockwise|*$Clockwise|c+BackSlash|c+\\']))

        items2 = ['|More peaks|SemiColon|;', '|Fewer peaks|s+SemiColon|:', None]
        for i in range(5):
            items2.append('|Higher peak {n}|{key}'.format(n=i+1, key='YUIOP'[i]))
            items2.append('|Lower peak {n}|{key}'.format(n=i+1, key='s+'+'YUIOP'[i]))
        # '@animal||', '#m|Field center', '#s|Field width', '#R|Space units', '#T|Time units', '#b|Kernel peaks', 
        self.menu.add_cascade(label='Params', menu=self.create_submenu(self.menu, [
            '|(Small adjust)||s+Q', '|Higher growth (m + 0.01)|Q', '|Lower growth (m - 0.01)|A',
            '|Wider growth (s + 0.001)|W', '|Narrower growth (s - 0.001)|S', None,
            '|*More states (P + 10)|E', '|*Fewer states (P - 10)|D', '|*Reset states|c+D', None,
            '|Zoom in space (R + 10)|R', '|Zoom out space (R - 10)|F',
            '|*Reset space|s+c+R', '|*Lifeform\'s original size|s+c+F', None,
            '|Slower time (T * 2)|T', '|Faster time (T / 2)|G', None,
            '|*Larger relative kernel (r + 0.1)|c+R', '|*Smaller relative kernel (r - 0.1)|c+F',
            '|*Larger relative increment (h + 0.1)|c+T', '|*Smaller relative increment (h - 0.1)|c+G', None,
            ('Peaks', items2)]))

        self.menu.add_cascade(label='Options', menu=self.create_submenu(self.menu, [
            '|*Search growth higher|c+Q', '|*Search growth lower|c+A', None,
            '@kn|Kernel core|c+Y', '@gn|Growth mapping|c+U', None,
            '@soft_clip|*Soft clip|c+I', '^automaton.is_inverted|*Invert mode|c+O', 
            '^automaton.is_arita_mode|*Target mode|c+P', None,  #'^automaton.is_multi_step|*Use multi-step|c+O',
            '@mask_rate|*Async rate|s+c+I', '@add_noise|*Noise rate|s+c+O', 
            '|*Reset async & noise|s+c+P']))

        self.menu.add_cascade(label='Stats', menu=self.create_submenu(self.menu, [
            '@markers_mode|Show marks|H', '^is_show_fps|*Show FPS|c+H', None,
            '@stats_mode|Show stats|J', '@stats_x|Stats X axis|K', '@stats_y|Stats Y axis|L', 
            '|*Show mass-growth|s+c+K', '|*Show trajectory|s+c+L', None,
            '|*Clear segment|c+J', '|*Clear all segments|s+c+J', 
            '@trim_segment|*Segment length|c+K', '^is_group_params|*Group by params|c+L']))

    def get_info_st(self):
        R, T, P = [self.world.model[k] for k in ('R', 'T', 'P')]
        P = str(P)
        if P == '0': P = '∞'
        status = 'EMP' if self.analyzer.is_empty else 'OVR' if self.analyzer.is_full else ''
        return "gen={}, t={}s, dt={}s, sampl={} {} obj={} | R={}, T={}, P={}".format(self.automaton.gen, self.automaton.time, 1/T, self.samp_freq, status, self.analyzer.object_num, R, T, P)

    def get_size_st(self):
        return "world={}, pixel={}".format("x".join(str(size) for size in SIZE), PIXEL)

    def get_time_st(self):
        T = self.world.model['T']
        status = 'EMP' if self.analyzer.is_empty else 'OVR' if self.analyzer.is_full else ''
        return "gen={}, t={}s, dt={}s, sampl={} {} obj={}".format(self.automaton.gen, self.automaton.time, 1/T, self.samp_freq, status, self.analyzer.object_num)

    def get_angular_st(self):
        if self.auto_rotate_mode in [3]:
            return "auto-rotate: {} axes={} sampl={} speed={:.2f}".format("clockwise" if self.is_samp_clockwise else "anti-clockwise", self.samp_sides, self.samp_gen, self.samp_rotate)
        else:
            return "not in auto-rotate mode"

    def get_kernel_st(self):
        st = [['*' if c0==c1 else '' for c1 in CHANNEL] for c0 in CHANNEL]
        for k in KERNEL:
            p = self.world.params[k]
            c0, c1 = p.get('c0', 0), p.get('c1', 0)
            st[c0][c1] += str(len(p['rings']))
        sizes = ' | '.join(','.join(a) for a in st)
        return "kernel sizes: {}".format(sizes)

    def update_info_bar(self):
        global STATUS
        if self.excess_key:
            #print(self.excess_key)
            self.excess_key = None
        if self.info_type or STATUS or self.is_show_fps:
            info_st = ""
            if STATUS: info_st = "\n".join(STATUS)
            elif self.is_show_fps and self.fps: info_st = "FPS: {:.1f}".format(self.fps)
            elif self.info_type == 'params': info_st = self.get_value_text('show_group') + " | " + self.world.params2st(self.world.params[self.show_kernel], is_brief=True); self.is_draw_params = True
            elif self.info_type == 'animal': info_st = self.world.long_name(); self.is_draw_params = True
            elif self.info_type == 'info': info_st = self.get_info_st()
            elif self.info_type == 'size': info_st = self.get_size_st()
            elif self.info_type == 'time': info_st = self.get_time_st()
            elif self.info_type == 'angular': info_st = self.get_angular_st()
            elif self.info_type == 'stats': info_st = "X axis: {xstat}, Y axis: {ystat}".format(xstat=self.analyzer.stats_fullname(i=self.stats_x), ystat=self.analyzer.stats_fullname(i=self.stats_y))
            elif self.info_type == 'slice': info_st = "slice: {slice}, Z axis: {d}th dim".format(slice=self.z_slices, d=DIM-self.z_axis)
            elif self.info_type == 'channel': info_st = "channel: {name}".format(name=self.show_which_channels_name())
            elif self.info_type == 'search': info_st = "auto find algorithm: {algo}".format(algo=self.get_value_text('search_algo'))
            elif self.info_type == 'kernel': info_st = self.get_kernel_st()
            elif self.info_type == 'object': info_st = "dist: {dist}, num: {num}".format(dist=self.analyzer.object_distance, num=self.analyzer.object_num)
            elif self.info_type in self.menu_values: info_st = "{text} [{value}]".format(text=self.VALUE_TEXT[self.info_type], value=self.get_value_text(self.info_type))
            self.info_bar.config(text=info_st)
            STATUS = []
            self.info_type = None
            if self.clear_job is not None:
                self.window.after_cancel(self.clear_job)
            self.clear_job = self.window.after(5000, self.clear_info)

    def clear_info(self):
        self.info_bar.config(text="")
        self.is_draw_params = False
        self.clear_job = None

    def loop(self):
        self.is_loop = True
        self.window.after(0, self.run)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.mainloop()

    def close(self):
        self.is_loop = False
        if self.recorder.is_recording:
            self.recorder.finish_record()
        self.window.destroy()

    def run(self):
        counter = 0
        while self.is_loop:
            counter += 1
            if self.last_key:
                self.process_key(self.last_key)
                self.last_key = None
            if self.is_closing:
                break
            if self.is_run:
                self.calc_fps()
                for automaton in self.automaton_list:
                    automaton.calc_once()
                # self.automaton.calc_once()
                self.analyzer.center_world()
                if self.show_what==4 or self.markers_mode in [1,3,5,7]:
                    if (self.search_mode != 0 and counter % self.samp_freq == 0) or \
                       (self.search_mode == 0 and self.automaton.gen % 10 == 0):
                        self.analyzer.detect_objects()
                self.analyzer.calc_stats(self.show_what, psd_x=self.stats_x, psd_y=self.stats_y, is_welch=True)
                self.analyzer.add_stats(psd_y=self.stats_y)
                if not self.is_layer_mode and not (self.search_mode == 0 and self.is_search_small):
                    self.back = None
                    self.clear_transform()
                if self.search_mode is not None:
                    self.do_search()
                if self.run_counter != -1:
                    self.run_counter -= 1
                    if self.run_counter == 0:
                        self.is_run = False
            else:
                self.finish_search()

            #if (self.automaton.gen % 10 == 0): print("{} {} {} {:.2f} {:.2f}".format(self.automaton.gen, self.analyzer.is_empty, self.analyzer.is_full, self.analyzer.mass, self.analyzer.border_sum))
            is_show_gen = (self.automaton.gen % 1000 == 0 and self.automaton.gen // 1000 > 3) #or (self.automaton.gen % 10 == 0 and (self.analyzer.is_full or self.analyzer.is_empty))
            if (self.search_mode != 0 and counter % self.samp_freq == 0) or \
               (self.search_mode == 0 and self.is_show_search):
                if not is_show_gen: self.update_info_bar()
                self.update_window()
                self.is_show_search = False
            if self.is_run: 
                if is_show_gen:
                    self.info_type = 'time'
                    self.update_info_bar()
                # if self.automaton.gen % 20 == 0:
                #     channel_alive_st = ', '.join('{}'.format(s) for s in self.analyzer.channel_alive)
                #     border_alive_st = ', '.join('{}'.format(s) for s in self.analyzer.border_alive)
                #     print("channel alive [", channel_alive_st, "] border alive [", border_alive_st, "]")

    def print_help(self):
        print("Lenia in n-Dimensions    by Bert Chan 2020    Run '{program} -h' for startup arguments.".format(program=sys.argv[0]))

if __name__ == '__main__':
    lenia = Lenia()
    #lenia.print_help()
    if CN==1 and KN==1:
        if lenia.ANIMAL_KEY_LIST != {} and lenia.ANIMAL_KEY_LIST['1'] != '':
            lenia.load_animal_code(lenia.world, lenia.ANIMAL_KEY_LIST['1'])
        else:
            lenia.world.model = {'R':DEF_R, 'T':10, 'kn':1, 'gn':1}
            #lenia.world.params = [{'b':[1], 'm':0.14, 's':0.014} for k in KERNEL]
            lenia.world.params = [{'rings':[DEFAULT_RING.copy()], 'm':0.14, 's':0.014, 'h':1, 'c0':0, 'c1':0} for k in KERNEL]
            lenia.automaton.calc_kernel()
            lenia.random_world()
    else:
        lenia.found_animal_id = 1
        lenia.key_press_internal('s+z')
    lenia.update_menu()
    lenia.loop()

''' for PyOpenCL in Windows:
install Intel OpenCL SDK
install Microsoft Visual C++ Build Tools
in Visual Studio Native Tools command prompt
> set INCLUDE=%INCLUDE%;%INTELOCLSDKROOT%include
> set LIB=%LIB%;%INTELOCLSDKROOT%lib\x86
> pip3 install pyopencl
'''
