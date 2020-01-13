import numpy as np                     # pip3 install numpy
import scipy.ndimage                   # pip3 install scipy
import scipy.spatial.distance, scipy.signal
import reikna.fft, reikna.cluda        # pip3 install pyopencl/pycuda, reikna
import PIL.Image, PIL.ImageTk          # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
try: import tkinter as tk
except: import Tkinter as tk
from fractions import Fraction
import copy, re, itertools, json, csv
import io, os, sys, argparse, datetime, time, subprocess, multiprocessing
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from scipy.ndimage.zoom()
warnings.filterwarnings('ignore', '.*divide by zero encountered.*')  # suppress warning from divide by zero in kernel_core()
warnings.filterwarnings('ignore', '.*invalid value encountered.*')  # suppress warning from divide by zero in normalize(), get_stat_row()
warnings.filterwarnings('ignore', '.*nperseg.*') # suppress warning from scipy.signal.periodogram/welch

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, 
description='''Lenia in n-Dimensions    by Bert Chan 2020

recommanded settings: (2D) -d2 -p2, (wide) -d2 -p0 -w 10 9, (3D) -d3 -p3, (4D) -d4 -p4''')
parser.add_argument('-d', '--dim', dest='D', default=2, action='store', type=int, help='number of dimensions (default 2D)')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-w', '--win', dest='W', default=[9], action='store', type=int, nargs='+', help='window size = 2^W (apply to all sides if only one value, default 2^9 = 512)')
group.add_argument('-s', '--size', dest='S', default=None, action='store', type=int, nargs='+', help='array size = 2^S (apply to all sides if only one value, default 2^(W-P) = 128)')
parser.add_argument('-p', '--pixel', dest='P', default=None, action='store', type=int, help='pixel size = 2^P (default 2^D)')
parser.add_argument('-b', '--border', dest='B', default=0, action='store', type=int, help='pixel border (default 0)')
args = parser.parse_args()

DIM = args.D
DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}
X_AXIS, Y_AXIS, Z_AXIS = -1, -2, -3

PIXEL_2 = args.P if args.P is not None else args.D
PIXEL_BORDER = args.B
if args.S is not None:
    SIZE_2 = args.S  # X, Y, Z, S...
else:
    SIZE_2 = [win_2 - PIXEL_2 for win_2 in args.W]
if len(SIZE_2) < DIM:
    SIZE_2 += [SIZE_2[-1]] * (DIM-len(SIZE_2))
# GoL 9,9,3,1   Lenia Lo 9,9,2,0  Hi 9,9,0,0   1<<9=512

SIZE = [1 << size_2 for size_2 in SIZE_2]
PIXEL = 1 << PIXEL_2
MID = [int(size / 2) for size in SIZE]
SIZEX, SIZEY = SIZE[0], SIZE[1]
MIDX, MIDY = MID[0], MID[1]

SIZER, SIZETH, SIZEF = min(MIDX, MIDY), SIZEX, MIDX
DEF_R   = int(np.power(2.0, min(SIZE_2) - 6) * DIM * 5)
RAND_R1 = int(np.power(2.0, min(SIZE_2) - 7) * DIM * 5)
RAND_R2 = int(np.power(2.0, min(SIZE_2) - 5) * DIM * 5)

EPSILON = 1e-10
ROUND = 10
STATUS = []
is_windows = (os.name == 'nt')
np.set_printoptions(precision=3)

class Board:
    def __init__(self, size=[0]*DIM):
        self.names = ['', '', '']
        self.params = {'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
        self.param_P = 0
        self.cells = np.zeros(size)

    @classmethod
    def from_values(cls, cells, params=None, names=None):
        self = cls()
        self.names = names.copy() if names is not None else None
        self.params = params.copy() if params is not None else None
        self.cells = cells.copy() if cells is not None else None
        return self

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
        self.params = data.get('params')
        if self.params:
            self.params = self.params.copy()
            self.params['b'] = Board.st2fracs(self.params['b'])
        self.cells = data.get('cells')
        if self.cells:
            if type(self.cells) in [tuple, list]:
                self.cells = ''.join(self.cells)
            self.cells = Board.rle2arr(self.cells)
        return self

    def to_data(self, is_shorten=True):
        rle_st = Board.arr2rle(self.cells, is_shorten)
        params2 = self.params.copy()
        params2['b'] = Board.fracs2st(params2['b'])
        data = {'code':self.names[0], 'name':self.names[1], 'cname':self.names[2], 'params':params2, 'cells':rle_st}
        return data

    def params2st(self):
        params2 = self.params.copy()
        params2['b'] = '[' + Board.fracs2st(params2['b']) + ']'
        return ','.join(['{}={}'.format(k,str(v)) for (k,v) in params2.items()])

    def long_name(self):
        # return ' | '.join(filter(None, self.names))
        return '{0} - {1} {2}'.format(*self.names)

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
    def arr2rle(A, is_shorten=True):
        values = np.rint(A*255).astype(int).tolist()  # [[255 255] [255 0]]
        if is_shorten:
            rle_groups = Board._recur_drill_list(0, values, lambda row: [( len(list(g)), Board.val2ch(v).strip() ) for v,g in itertools.groupby(row)] )
            st = Board._recur_join_st(0, rle_groups, lambda row: [(str(n) if n>1 else '')+c for n,c in row] )  # "2 yO $ 1 yO"
        else:
            st = Board._recur_join_st(0, values, lambda row: [Board.val2ch(v) for v in row] )
        return st + '!'

    @staticmethod
    def rle2arr(st):
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
                    #print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
                last, count = '', ''
        A = stacks[DIM-1]
        max_lens = [0 for dim in range(DIM)]
        Board._recur_get_max_lens(0, A, max_lens)
        Board._recur_cubify(0, A, max_lens)
        return np.asarray(A)

    @staticmethod
    def fracs2st(B):
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        return [Fraction(st) for st in st.split(',')]

    def clear(self):
        self.cells.fill(0)

    def _recur_add(self, dim, cells1, cells2, shift, is_centered):
        size1, size2 = cells1.shape[0], cells2.shape[0]
        size0 = min(size1, size2)
        start1 = (size1 - size0)//2 + shift[dim] if is_centered else shift[dim]
        start2 = (size2 - size0)//2 if is_centered else 0
        if dim < DIM-1:
            for x in range(size0):
                self._recur_add(dim+1, cells1[(start1+x)%size1], cells2[start2+x], shift, is_centered)
        else:
            for x in range(size0):
                if cells2[start2+x] > EPSILON:
                    cells1[(start1+x)%size1] = cells2[start2+x]

    def add(self, part, shift=[0]*DIM, is_centered=True):
        # shift: s, z, y, x
        # assert self.params['R'] == part.params['R']
        self._recur_add(0, self.cells, part.cells, shift, is_centered)
        return self

    def transform(self, tx, mode='RZSF', z_axis=Z_AXIS, is_world=False):
        if 'R' in mode and tx['rotate'] != [0]*3:
            if DIM == 2:
                self.cells = scipy.ndimage.rotate(self.cells, -tx['rotate'][1], reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')
            elif DIM >= 3:
                self.cells = scipy.ndimage.rotate(self.cells, tx['rotate'][2], axes=(X_AXIS, z_axis), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by y axis = x-z plane
                self.cells = scipy.ndimage.rotate(self.cells, tx['rotate'][1], axes=(z_axis, Y_AXIS), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by x axis = z-y plane
                self.cells = scipy.ndimage.rotate(self.cells, tx['rotate'][0], axes=(Y_AXIS, X_AXIS), reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')  # rotate by z axis = y-x plane
        if 'Z' in mode and tx['R'] != self.params['R']:
            # print('* {} / {}'.format(tx['R'], self.params['R']))
            shape_orig = self.cells.shape
            self.cells = scipy.ndimage.zoom(self.cells, tx['R'] / self.params['R'], order=0)
            if is_world:
                self.cells = Board(shape_orig).add(self).cells
            self.params['R'] = tx['R']
        if 'F' in mode and tx['flip'] != -1:
            extra_slice = [slice(None)] * (DIM-2)
            if tx['flip'] in [0,1]: self.cells = np.flip(self.cells, axis=DIM-2+tx['flip'])
            elif tx['flip'] == 2: slice1 = [slice(None), slice(None,-MIDX-1,-1)]; slice2 = [slice(None), slice(None,MIDX)]; self.cells[extra_slice + slice1] = self.cells[extra_slice + slice2]
            elif tx['flip'] == 3: slice1 = [slice(None), slice(None,-MIDX-1,-1)]; slice2 = [slice(None,None,-1), slice(None,MIDX)]; self.cells[extra_slice + slice1] = self.cells[extra_slice + slice2]
            elif tx['flip'] == 4: i_upper = np.triu_indices(SIZEX, -1); self.cells[i_upper] = self.cells.T[i_upper]
        if 'S' in mode and tx['shift'] != [0]*DIM:
            self.cells = scipy.ndimage.shift(self.cells, tx['shift'], order=0, mode='wrap')
            # self.cells = np.roll(self.cells, tx['shift'], (2, 1, 0))
        return self

    def add_transformed(self, part, tx):
        part = copy.deepcopy(part)
        self.add(part.transform(tx, mode='RZF'), tx['shift'])
        return self

    def crop(self):
        #vmin = np.amin(self.cells)
        coords = np.argwhere(self.cells > EPSILON)
        if coords.size == 0:
            self.cells = np.zeros([1]*DIM)
        else:
            min_point = coords.min(axis=0)
            max_point = coords.max(axis=0) + 1
            slices = [slice(x1, x2) for x1, x2 in zip(min_point, max_point)]
            self.cells = self.cells[tuple(slices)]
        return self

    def restore_to(self, dest):
        dest.params = self.params.copy()
        dest.cells = self.cells.copy()
        dest.names = self.names.copy()

class Automaton:
    kernel_core = {
        0: lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
        1: lambda r: np.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        3: lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
    }
    growth_func = {
        0: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        1: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (np.abs(n-m)<=s) * 2 - 1  # step (stpz)
    }

    def __init__(self, world):
        self.world = world
        self.world_FFT = np.zeros(world.cells.shape)
        self.potential_FFT = np.zeros(world.cells.shape)
        self.potential = np.zeros(world.cells.shape)
        self.field = np.zeros(world.cells.shape)
        self.field_old = None
        self.change = np.zeros(world.cells.shape)
        # self.deconv = np.zeros(world.cells.shape)
        self.X = [None]*DIM
        self.D = None
        self.Z_depth = None
        self.TH = None
        self.R = None
        self.polar_X = None
        self.polar_Y = None
        self.gen = 0
        self.time = 0
        self.is_multi_step = False
        self.is_soft_clip = False
        self.mask_rate = 0
        self.add_noise = 0
        self.is_inverted = False
        self.is_gpu = True
        self.has_gpu = True
        self.compile_gpu(self.world.cells)
        self.calc_kernel()

    def kernel_shell(self, r):
        B = len(self.world.params['b'])
        Br = B * r
        bs = np.asarray([float(f) for f in self.world.params['b']])
        b = bs[np.minimum(np.floor(Br).astype(int), B-1)]
        kfunc = Automaton.kernel_core[self.world.params.get('kn') - 1]
        return (r<1) * kfunc(np.minimum(Br % 1, 1)) * b

    @staticmethod
    def soft_max(x, m, k):
        ''' Soft maximum: https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
        return np.log(np.exp(k*x) + np.exp(k*m)) / k

    @staticmethod
    def soft_clip(x, min, max, k):
        a = np.exp(k*x)
        b = np.exp(k*min)
        c = np.exp(-k*max)
        return np.log( 1/(a+b)+c ) / -k
        # return Automaton.soft_max(Automaton.soft_max(x, min, k), max, -k)

    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft1 = self.gpu_fftn = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create()
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
    def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)

    def calc_once(self, is_update=True):
        A = self.world.cells
        dt = 1 / self.world.params['T']
        self.world_FFT = self.fftn(A)
        self.potential_FFT = self.kernel_FFT * self.world_FFT
        self.potential = self.fftshift(np.real(self.ifftn(self.potential_FFT)))
        gfunc = Automaton.growth_func[self.world.params.get('gn') - 1]

        # self.deconv_FFT = self.world_FFT / self.kernel_FFT
        # self.deconv = self.fftshift(np.abs(self.ifftn(1/self.deconv_FFT))) * self.kernel_sum * 50
        # self.deconv_FFT = self.fftn(self.deconv) * self.kernel_FFT
        # self.deconv = self.fftshift(np.real(self.ifftn(self.deconv_FFT)))

        #m = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['m']
        #s = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['s']
        m, s = self.world.params['m'], self.world.params['s']
        self.field = gfunc(self.potential, m, s)
        if self.is_multi_step and self.field_old:
            D = 1/2 * (3 * self.field - self.field_old)
            self.field_old = self.field.copy()
        else:
            D = self.field
        A_new = A + dt * D
        if self.add_noise > 0:
            rand = (np.random.random_sample(A_new.shape) - 0.5) * (self.add_noise/10) + 1
            A_new *= rand
        if self.is_soft_clip:
            A_new = Automaton.soft_clip(A_new, 0, 1, 1/dt)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
        else:
            A_new = np.clip(A_new, 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
        if self.world.param_P > 0:
            A_new = np.around(A_new * self.world.param_P) / self.world.param_P
        self.change = (A_new - A) / dt
        if is_update:
            if self.mask_rate > 0:
                mask = np.random.random_sample(A_new.shape) > (self.mask_rate/10)
                self.world.cells[mask] = A_new[mask]
            else:
                self.world.cells = A_new
            self.gen += 1
            self.time = round(self.time + dt, ROUND)
        if self.is_gpu:
            self.gpu_thr.synchronize()

    def calc_kernel(self):
        R = self.world.params['R']
        dims = [slice(0, size) for size in SIZE]
        I = list(reversed(np.mgrid[list(reversed(dims))]))  # I, J, K, L
        self.X = [(i - mid) / R for i, mid in zip(I, MID)]  # X, Y, Z, S
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

        self.kernel = self.kernel_shell(self.D)
        self.kernel_sum = self.kernel.sum()
        kernel_norm = self.kernel / self.kernel_sum
        self.kernel_FFT = self.fftn(kernel_norm)
        self.kernel_updated = False

    def reset(self):
        self.gen = 0
        self.time = 0
        self.field_old = None

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
    SEGMENT_LEN = 512
    PSD_INTERVAL = 32
    def get_stat_row(self):
        R, T, pm, ps = [self.world.params[k] for k in ('R', 'T', 'm', 's')]
        if self.m_center is not None:
            pos = self.m_center * R + self.total_shift_idx
        else:
            pos = [0]*DIM
        RN = np.power(R, DIM)
        return [pm, ps, self.automaton.gen, self.automaton.time, 
                self.mass/RN, self.growth/RN, np.sqrt(self.inertia/self.mass),  # self.inertia/RN  # self.inertia*RN, 
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
        self.is_trim_segment = True
        self.is_calc_symmetry = False
        self.is_calc_psd = False
        self.reset()

    def reset(self):
        self.reset_values()
        self.reset_last()
        self.reset_position()
        self.reset_polar()
        self.clear_series()

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

        R, T = [self.world.params[k] for k in ('R', 'T')]
        A = self.world.cells
        G = np.maximum(self.automaton.field, 0)
        X = self.automaton.X
        m0 = self.mass = A.sum()
        g0 = self.growth = G.sum()
        self.is_empty = (self.mass < EPSILON)
        border_sum = 0
        for d in range(DIM):
            slices = [0 if d==d2 else slice(None) for d2 in range(DIM)]
            border_sum += A[tuple(slices)].sum()
            slices = [A.shape[d]-1 if d==d2 else slice(None) for d2 in range(DIM)]
            border_sum += A[tuple(slices)].sum()
        self.is_full = (border_sum > 0)

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
            
            self.lyapunov += ( np.log(abs(self.automaton.change.sum())) - self.lyapunov ) / self.automaton.gen

            if polar_what==0: A2 = self.world.cells
            elif polar_what==1: A2 = self.automaton.potential
            elif polar_what==2: A2 = self.automaton.field
            else: A2 = self.world.cells

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

    def stat_name(self, i=None, x=None):
        if not x: x = self.STAT_HEADERS[i]
        return '{0}={1}'.format(x, self.STAT_NAMES[x])

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
            self.series[-1] = [[self.world.params['m'], self.world.params['s']] + [np.nan] * (len(self.STAT_HEADERS)-2)]
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
        multi = max(1, self.world.params['T'] // 10)
        if self.series == []:
            self.new_segment()
        segment = self.series[-1]
        self.current = self.get_stat_row()
        segment.append(self.current)
        if self.polar_R is not None:
            self.series_R.append(self.polar_R)
            self.series_TH.append(self.polar_TH)
        if self.is_trim_segment:
            if self.automaton.gen <= self.SEGMENT_INIT * multi:
                limit = self.SEGMENT_INIT_LEN * multi
            else:
                limit = self.SEGMENT_LEN * multi
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
        self.last_shift_idx = (self.m_center * self.world.params['R']).astype(int)
        self.total_shift_idx += self.last_shift_idx
        self.world.cells = np.roll(self.world.cells, -self.last_shift_idx, axes)
        self.automaton.potential = np.roll(self.automaton.potential, -self.last_shift_idx, axes)
        self.automaton.field = np.roll(self.automaton.field, -self.last_shift_idx, axes)
        if self.automaton.field_old is not None:
            self.automaton.field_old = np.roll(self.automaton.field_old, -self.last_shift_idx, axes)
        self.automaton.change = np.roll(self.automaton.change, -self.last_shift_idx, axes)
        # self.world.cells = scipy.ndimage.shift(self.world.cells, -self.last_shift_idx, order=0, mode='wrap')

class Recorder:
    RECORD_ROOT = 'record'
    FRAME_EXT = '.png'
    VIDEO_EXT = '.mov'
    GIF_EXT = '.gif'
    ANIM_FPS = 25
    ffmpeg_cmd = ['/usr/local/bin/ffmpeg',
        '-loglevel','warning', '-y',  # glocal options
        '-f','rawvideo', '-vcodec','rawvideo', '-pix_fmt','rgb24',  # input options
        '-s','{}x{}'.format(SIZEX*PIXEL, SIZEY*PIXEL), '-r',str(ANIM_FPS),
        '-i','{input}',  # input pipe
        # '-an', '-vcodec','h264', '-pix_fmt','yuv420p', '-crf','1',  # output options
        '-an', '-vcodec','copy',  # output options
        '{output}']  # ouput file

    def __init__(self, world):
        self.world = world
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
        else:
            self.finish_record()

    def start_record(self):
        global STATUS
        ''' https://trac.ffmpeg.org/wiki/Encode/H.264
            https://trac.ffmpeg.org/wiki/Slideshow '''
        self.is_recording = True
        STATUS.append("> start " + ("saving frames" if self.is_save_frames else "recording video") + " and GIF...")
        self.record_id = '{}-{}'.format(self.world.names[0].split('(')[0], datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        self.record_seq = 1
        self.video_path = os.path.join(self.RECORD_ROOT, self.record_id + self.VIDEO_EXT)
        self.gif_path = os.path.join(self.RECORD_ROOT, self.record_id + self.GIF_EXT)
        self.img_dir = os.path.join(self.RECORD_ROOT, self.record_id)
        if self.is_save_frames:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        else:
            cmd = [s.replace('{input}', '-').replace('{output}', self.video_path) for s in self.ffmpeg_cmd]
            try:
                self.video = subprocess.Popen(cmd, stdin=subprocess.PIPE)    # stderr=subprocess.PIPE
            except FileNotFoundError as e:
                self.video = None
                STATUS.append("> no ffmpeg program found!")
        self.gif = []

    def save_image(self, img, filename=None):
        self.record_id = '{}-{}'.format(self.world.names[0].split('(')[0], datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
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
        durations = [1000 // self.ANIM_FPS] * len(self.gif)
        durations[-1] *= 10
        self.gif[0].save(self.gif_path, format=self.GIF_EXT.lstrip('.'), save_all=True, append_images=self.gif[1:], loop=0, duration=durations)
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
    FOUND_ANIMALS_PATH = 'found.json'

    def __init__(self):
        self.is_run = True
        self.run_counter = -1
        self.is_closing = False
        self.is_advanced_menu = False

        self.show_what = 0
        self.polar_mode = 0
        self.markers_mode = 0

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
        self.is_layered = False
        self.is_auto_load = False
        self.last_seed = None
        self.backup_world = None
        self.backup_automaton = None
        self.backup_analyzer = None

        self.search_mode = None
        self.is_search_small = False
        self.is_show_search = False
        self.search_total = 0
        self.search_success = 0
        
        self.is_show_slice = False
        self.z_slices = [MID[DIM-1-d] for d in range(DIM-2)]  # S, Z
        self.z_axis = DIM-3

        ''' http://hslpicker.com/ '''
        self.colormaps = [
            self.create_colormap(np.asarray([[0,0,4],[0,0,8],[0,4,8],[0,8,8],[4,8,4],[8,8,0],[8,4,0],[8,0,0],[4,0,0]])), #BCYR
            self.create_colormap(np.asarray([[7,6,7],[5,4,5],[4,1,4],[1,3,6],[3,4,6],[4,5,7],[2,6,3],[5,6,4],[6,7,5],[8,8,3],[8,6,2],[8,5,1],[7,0,0]])), #Paul Tol's Rainbow
            self.create_colormap(np.asarray([[0,2,0],[0,4,0],[4,6,0],[8,8,0],[8,4,4],[8,0,8],[4,0,8],[0,0,8],[0,0,4]])), #GYPB
            self.create_colormap(np.asarray([[4,0,2],[8,0,4],[8,0,6],[8,0,8],[4,4,4],[0,8,0],[0,6,0],[0,4,0],[0,2,0]])), #PPGG
            self.create_colormap(np.asarray([[4,4,6],[2,2,4],[2,4,2],[4,6,4],[6,6,4],[4,2,2]])), #BGYR
            self.create_colormap(np.asarray([[4,6,4],[2,4,2],[4,4,2],[6,6,4],[6,4,6],[2,2,4]])), #GYPB
            self.create_colormap(np.asarray([[6,6,4],[4,4,2],[4,2,4],[6,4,6],[4,6,6],[2,4,2]])), #YPCG
            #self.create_colormap(np.asarray([[0,0,0],[3,3,3],[4,4,4],[5,5,5],[8,8,8]]))] #B/W
            self.create_colormap(np.asarray([[8,8,8],[7,7,7],[5,5,5],[3,3,3],[0,0,0]]), is_marker_w=False), #W/B
            self.create_colormap(np.asarray([[0,0,0],[3,3,3],[5,5,5],[7,7,7],[8,8,8]]))] #B/W
        self.colormap_id = 0

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
        self.animal_data = None
        self.found_animal_data = None
        self.read_animals()
        self.read_found_animals()
        self.world = Board(list(reversed(SIZE)))
        # self.target = Board((SIZEY, SIZEX))
        self.automaton = Automaton(self.world)
        self.analyzer = Analyzer(self.automaton)
        self.recorder = Recorder(self.world)
        self.clear_transform()
        self.create_window()
        self.create_menu()
        #self.cppn = CPPN(self.automaton.X, self.automaton.Y, self.automaton.Z, self.automaton.S, self.automaton.D, z_size=8, scale=2, net_size=64)
        #self.font = PIL.ImageFont.truetype('resource/monaco.ttf', 10)
        #self.convert_font_run_once('resource/bitocra-13.bdf')
        self.font = PIL.ImageFont.load('resource/bitocra-13.pil')

    def convert_font_run_once(self, font_file_path):
        import PIL.BdfFontFile, PIL.PcfFontFile
        ''' https://stackoverflow.com/questions/48304078/python-pillow-and-font-conversion '''
        ''' https://github.com/ninjaaron/bitocra '''
        with open(font_file_path, 'rb') as fp:
            p = PIL.BdfFontFile.BdfFontFile(fp) #PcfFontFile if you're reading PCF files
            p.save(font_file_path)

    def clear_transform(self):
        self.tx = {'shift':[0]*DIM, 'rotate':[0]*3, 'R':self.world.params['R'], 'flip':-1}

    def read_animals(self):
        is_replace = self.animal_data is not None
        try:
            with open(self.ANIMALS_PATH, 'r', encoding='utf-8') as file:
                new_animal_data = json.load(file)
                self.animal_data = new_animal_data
                if is_replace:
                    STATUS.append("> lifeforms reloaded")
        except IOError:
            pass
        except json.JSONDecodeError as e:
            STATUS.append("> JSON file error")
            print(e)

    def read_found_animals(self):
        try:
            with open(self.FOUND_ANIMALS_PATH, 'r', encoding='utf-8') as file:
                st = file.read()
                st = "[" + st[:-2] + "]"
                new_found_animal_data = json.loads(st)
                self.found_animal_data = new_found_animal_data
        except IOError:
            pass
        except json.JSONDecodeError as e:
            print(e)

    def load_animal_id(self, world, id, **kwargs):
        if self.animal_data is None or self.animal_data == []:
            return
        self.animal_id = max(0, min(len(self.animal_data)-1, id))
        self.load_part(world, Board.from_data(self.animal_data[self.animal_id]), **kwargs)

    def load_found_animal_id(self, world, id, **kwargs):
        if self.found_animal_data is None or self.found_animal_data == []:
            return
        self.found_animal_id = max(0, min(len(self.found_animal_data)-1, id))
        self.load_part(world, Board.from_data(self.found_animal_data[self.found_animal_id]), is_use_part_R=True, **kwargs)
        self.world.names = ['Found #' + str(self.found_animal_id + 1), '', '']

    def load_animal_code(self, world, code, **kwargs):
        if self.animal_data is None or self.animal_data == []:
            return
        if not code: return
        id = self.get_animal_id(code)
        if id is not None and id != -1:
            self.load_animal_id(world, id, **kwargs)

    def get_animal_id(self, code):
        if self.animal_data is None or self.animal_data == []:
            return -1
        code_sp = code.split(':')
        n = int(code_sp[1]) if len(code_sp)==2 else 1
        itr = (id for (id, data) in enumerate(self.animal_data) if data["code"]==code_sp[0])
        for i in range(n):
            id = next(itr, None)
        return id

    def search_animal_id(self, prefix, old_id, dir):
        if self.animal_data is None or self.animal_data == []:
            return
        id = old_id + dir
        while id >= 0 and id < len(self.animal_data):
            if self.animal_data[id]["name"].startswith(prefix):
                return id
            else:
                id += dir
        return old_id

    def search_animal(self, world, prefix, dir):
        if self.animal_data is None or self.animal_data == []:
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

    def load_part(self, world, part, is_replace=True, is_use_part_R=False, is_random=False, is_auto_load=False, repeat=1):
        if part is None:
            return
        self.fore = part
        if part.names is not None and part.names[0].startswith('~'):
            part.names[0] = part.names[0].lstrip('~')
            world.params['R'] = part.params['R']
            self.automaton.calc_kernel()
        if part.names is not None and is_replace:
            world.names = part.names.copy()
        if part.cells is not None:
            if part.params is None:
                part.params = world.params
            is_life = (world.params.get('kn') == 4)
            will_be_life = (part.params.get('kn') == 4)
            if not is_life and will_be_life:
                self.colormap_id = len(self.colormaps) - 1
                self.window.title('Conway\'s Game of Life')
            elif is_life and not will_be_life:
                self.colormap_id = 0
                world.params['R'] = DEF_R
                self.automaton.calc_kernel()
                self.window.title('Lenia {0}D'.format(DIM))
            if self.is_layered:
                self.back = copy.deepcopy(world)
            if is_replace and not self.is_layered:
                if not is_auto_load:
                    if is_use_part_R:
                        R = part.params['R']
                    else:
                        R = world.params['R']
                    world.params = {**part.params, 'R':R}
                    self.automaton.calc_kernel()
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
                    self.tx['rotate'] = (np.random.rand(3) * 360).tolist()
                    shape1 = world.cells.shape
                    shape0 = min(part.cells.shape, world.cells.shape)
                    self.tx['shift'] = [np.random.randint(d1 + d0) - d1//2 for d0, d1 in zip(shape0, shape1)]
                    self.tx['flip'] = np.random.randint(3) - 1
                world.add_transformed(part, self.tx)

    def check_auto_load(self):
        if self.is_auto_load:
            self.load_part(self.world, self.fore, is_auto_load=True)
        else:
            self.automaton.reset()

    def transform_world(self):
        if self.is_layered:
            if self.back is not None:
                self.world.cells = self.back.cells.copy()
                self.world.params = self.back.params.copy()
                self.world.transform(self.tx, mode='Z', z_axis=self.z_axis, is_world=True)
                self.world.add_transformed(self.fore, self.tx)
        else:
            if not self.is_run:
                if self.back is None:
                    self.back = copy.deepcopy(self.world)
                else:
                    self.world.cells = self.back.cells.copy()
                    self.world.params = self.back.params.copy()
            if self.tx['flip'] < 2:
                self.world.transform(self.tx, z_axis=self.z_axis, is_world=True)
            else:
                self.world.transform(self.tx, mode='RS', z_axis=self.z_axis, is_world=True)
                self.world.transform(self.tx, mode='ZF', z_axis=self.z_axis, is_world=True)
        self.automaton.calc_kernel()
        self.analyzer.reset_last()

    def world_updated(self):
        if self.is_layered:
            self.back = copy.deepcopy(self.world)
        self.automaton.reset()
        self.analyzer.reset()

    def clear_world(self):
        self.world.clear()
        self.world_updated()
        self.world.names = ['', '', '']

    def random_world(self, is_reseed=False, is_fill=False):
        R = self.world.params['R']
        if is_reseed and self.last_seed is not None:
            np.random.set_state(self.last_seed)
        else:
            self.last_seed = np.random.get_state()

        if is_fill:
            dims = [size - R*2 for size in SIZE]
            rand = np.random.random_sample(tuple(reversed(dims))) * 0.9
            self.world.clear()
            self.world.add(Board.from_values(rand))
        else:
            self.world.clear()
            dim = int(R * 0.9)
            dims = [dim]*DIM
            for i in range(np.random.randint(15, 40)):
                rand = np.random.random_sample(tuple(reversed(dims))) * 0.9
                border = R//2 + dim//2
                shift = [np.random.randint(border, size - border) - size//2 if border < size - border else 0  for size in SIZE]
                self.world.add(Board.from_values(rand), shift=list(reversed(shift)))

        self.world_updated()
        self.world.names = ['', '', '']
 
    def random_world_and_params(self, is_reseed=False, is_fill=False):
        p = self.world.params
        # p['R'] = np.random.randint(15, 35)
        p['R'] = np.random.randint(RAND_R1, RAND_R2) if RAND_R1 < RAND_R2 else RAND_R1
        B = np.random.randint(1, 4)
        p['b'] = [Fraction(np.random.randint(0, 12), 12) for i in range(B)]
        p['b'][np.random.randint(B)] = 1
        p['m'] = round(np.random.rand() * (0.5-0.1) + 0.1, 2)  # 0.1 .. 0.5
        # p['s'] = round(np.random.rand() * (0.05-0.01) + 0.01, 3)  # 0.01 .. 0.05
        d = np.random.rand() * (20.0-5.0) + 5.0
        p['s'] = round(p['m'] / d, 3)  # m/5.0 .. m/20.0
        self.automaton.calc_kernel()
        self.random_world(is_reseed=is_reseed, is_fill=is_fill)

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

    def toggle_search(self, dir, small=False):
        if self.search_mode is None:
            self.search_mode = dir
            self.is_search_small = small
            if dir == 0:
                self.is_show_search = False
                self.search_total = 0
                self.search_success = 0
                self.stats_mode = 1
                self.stats_x = 3  #t
                self.stats_y = 4  #m
            else:
                self.is_auto_center = True
                self.is_auto_load = True
        else:
            self.stop_search()

    def stop_search(self):
        self.search_mode = None
        if self.search_mode == 0:
            self.read_found_animals()

    def do_search(self):
        global STATUS
        s = 's+' if self.is_search_small else ''
        if self.search_mode == +1:
            if self.analyzer.is_empty: self.key_press_internal(s+'w')
            elif self.analyzer.is_full: self.key_press_internal(s+'q')
        elif self.search_mode == -1:
            if self.analyzer.is_empty: self.key_press_internal(s+'a')
            elif self.analyzer.is_full: self.key_press_internal(s+'s')
        elif self.search_mode == 0:
            if self.analyzer.is_empty or self.analyzer.is_full:
                self.key_press_internal('m')
                self.is_show_search = True
                self.search_total += 1
            elif self.automaton.time >= 25.0:
                self.append_found_file()
                self.key_press_internal('m')
                self.is_show_search = True
                self.search_total += 1
                self.search_success += 1
                #print('Found:', self.world.params2st())
            if self.is_show_search:
                STATUS.append("{0} found in {1} trials, saved to {2}".format(self.search_success, self.search_total, self.FOUND_ANIMALS_PATH))

    def append_found_file(self):
        A = copy.deepcopy(self.world)
        A.crop()
        data = A.to_data()
        self.found_animal_data.append(data)
        try:
            with open(self.FOUND_ANIMALS_PATH, 'a+', encoding='utf-8') as file:
                st = json.dumps(data, separators=(',', ':'), ensure_ascii=False) + ',\n'
                file.write(st)
        except IOError as e:
            STATUS.append("I/O error({0}): {1}".format(e.errno, e.strerror))

    def create_window(self):
        self.window = tk.Tk()
        self.window.title('Lenia {0}D'.format(DIM))
        icon = tk.Image("photo", file="resource/icon2.png")
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

    def update_window(self, show_arr=None, is_reimage=True):
        if is_reimage:
            if self.show_what==6:
                xgrad, ygrad = np.gradient(self.automaton.potential)
                grad = np.sqrt(xgrad**2 + ygrad**2) * self.world.params['R'] * 1.5
                #print(np.min(grad), np.max(grad))

            if show_arr is not None:
                self.draw_world(show_arr, 0, 1)
            elif self.stats_mode in [0, 1, 2, 5]:
                change_range = 1 if not self.automaton.is_soft_clip else 1.4
                if self.show_what==0: self.draw_world(self.world.cells, 0, 1, is_shift=True, is_shift_zero=True, markers=['world','arrow','scale','grid','colormap','params'])
                # if self.show_what==0: self.draw_world(self.analyzer.aaa, 0, 1, is_shift=True, is_shift_zero=True, markers=['world','arrow','scale','grid','colormap','params'])
                elif self.show_what==1: self.draw_world(self.automaton.potential, 0, 2*self.world.params['m'], is_shift=True, is_shift_zero=True, markers=['arrow','scale','grid','colormap','params'])
                elif self.show_what==2: self.draw_world(self.automaton.field, -1, 1, is_shift=True, markers=['arrow','scale','grid','colormap','params'])
                # elif self.show_what==2: self.draw_world(self.automaton.deconv, 0, 1, is_shift=True, markers=['arrow','scale','grid','colormap','params'])
                elif self.show_what==3: self.draw_world(self.automaton.kernel, 0, 1, markers=['scale','fixgrid','colormap','params'])
                elif self.show_what==4: self.draw_world(self.automaton.fftshift(np.log(np.abs(self.automaton.world_FFT))), 0, 5, markers=['colormap','params'])  #-10, 10
                elif self.show_what==5: self.draw_world(self.automaton.fftshift(np.log(np.abs(self.automaton.potential_FFT))), -20, 5, markers=['colormap','params'])  #-40, 10
                elif self.show_what==6: self.draw_world(grad, 0, 1, is_shift=True, markers=['arrow','scale','grid','colormap','params'])
                elif self.show_what==7: self.draw_world(self.automaton.change, -change_range, change_range, is_shift=True, markers=['arrow','scale','grid','colormap','params'])
            elif self.stats_mode in [3, 4]:
                self.draw_black()
            elif self.stats_mode in [6]:
                self.draw_recurrence()

            if self.stats_mode in [1, 2, 3, 4]:
                self.draw_stats(is_current_series=self.stats_mode in [1, 2, 3], is_small=self.stats_mode in [1])
            elif self.stats_mode in [5]:
                self.draw_psd(is_welch=True)

            if self.recorder.is_recording:  # and self.is_run:
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

    def get_image(self, buffer):
        y, x = buffer.shape
        buffer = np.repeat(buffer, PIXEL, axis=0)
        buffer = np.repeat(buffer, PIXEL, axis=1)
        # zero = np.uint8(np.clip(self.normalize(0, vmin, vmax), 0, 1) * 252)
        zero = 0
        for i in range(PIXEL_BORDER):
            buffer[i::PIXEL, :] = zero;  buffer[:, i::PIXEL] = zero
        return PIL.Image.frombuffer('P', (x*PIXEL,y*PIXEL), buffer, 'raw', 'P', 0, 1)

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

    def draw_world(self, A, vmin=0, vmax=1, is_shift=False, is_shift_zero=False, markers=[]):
        R = self.world.params['R']
        axes = tuple(reversed(range(DIM)))
        if is_shift and not self.is_auto_center:
            shift = self.analyzer.total_shift_idx #if 'world' in markers else self.analyzer.total_shift_idx - self.analyzer.last_shift_idx 
            A = np.roll(A, shift.astype(int), axes)
            # A = scipy.ndimage.shift(A, self.analyzer.total_shift_idx, order=0, mode='wrap')
        if is_shift_zero and self.automaton.is_soft_clip:
            if vmin==0: vmin = np.amin(A)

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
                        A = A[self.z_slices[d]]
                else:
                    for d in range(DIM-3):
                        A = A[self.z_slices[d]]
                    #auto-spin
                    #A = scipy.ndimage.rotate(A, (self.automaton.gen*2) % 360, axes=(X_AXIS, self.z_axis), reshape=False, order=1, mode='wrap')
                    A = (A * self.automaton.Z_depth).sum(axis=0)

            buffer = np.uint8(np.clip(self.normalize(A, vmin, vmax), 0, 1) * 252)  # .copy(order='C')
            self.draw_grid(buffer, markers, is_fixed='fixgrid' in markers)
            # self.draw_params(buffer, markers)
            self.img = self.get_image(buffer)
            self.draw_arrow(markers)
            self.draw_symmetry(markers)
            if PIXEL > 1 and self.is_auto_center and is_shift and self.analyzer.m_center is not None:
                m1 = self.analyzer.m_center * R * PIXEL
                self.shift_img(self.img, int(m1[0]), int(m1[1]), is_rotate=False)
            if angle_shift != 0:
                self.img = self.img.rotate(-angle_shift*360, resample=PIL.Image.NEAREST, expand=False)
                # samp_rotate = OG:96, D7:-8, D8:-6, D9:-5.4
            self.draw_symmetry_title(markers)
            self.draw_legend(markers)
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
                    A2[:SIZER, -2:] = Y[:] / Y_max
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
            self.img = self.get_image(buffer)
            self.draw_arrow(markers)
            self.draw_symmetry(markers)
            if self.polar_mode in [2] and angle_shift != 0:
                dx = int((-angle_shift % 1)*SIZETH*PIXEL)
                self.shift_img(self.img, dx, 0, is_rotate=True)
            self.draw_symmetry_title(markers)
            self.draw_legend(markers)

        self.img.putpalette(self.colormaps[self.colormap_id])

    def draw_title(self, draw, line, title, color=255):
        title_w, title_h = draw.textsize(title)
        title_x, title_y = MIDX*PIXEL-title_w//2, line*12+7
        draw.text((title_x,title_y), title, fill=color, font=self.font)

    # def draw_histo(self, A, vmin, vmax):
        # draw = PIL.ImageDraw.Draw(self.img)
        # HWIDTH = 1
        # hist, _ = np.histogram(A, bins=SIZEX//HWIDTH, range=(vmin,vmax))  #, density=True)
        # #print(vmin, vmax, A.min(), A.max())
        # for i in range(hist.shape[0]):
            # h = hist[i]
            # y = h  #(h*SIZEY).astype(int)
            # draw.rectangle([(i*HWIDTH*PIXEL,SIZEY*PIXEL),((i+1)*HWIDTH*PIXEL-1,(SIZEY-y)*PIXEL)], fill=254)
        # del draw

    def draw_black(self):
        isize = (SIZEX*PIXEL,SIZEY*PIXEL)
        asize = (SIZEY*PIXEL,SIZEX*PIXEL)
        self.img = PIL.Image.frombuffer('L', isize, np.zeros(asize), 'raw', 'L', 0, 1)

    def draw_grid(self, buffer, markers=[], is_fixed=False):
        R = self.world.params['R']
        n = R // 40 if R >= 15 else -1
        if ('grid' in markers or 'fixgrid' in markers) and self.markers_mode in [0,1,2,3]:
            for i in range(-n, n+1):
                sx, sy = 0, 0
                if self.is_auto_center and not is_fixed:
                    sx, sy, *_ = self.analyzer.total_shift_idx.astype(int)
                grid = buffer[(MIDY - sy + i) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R];  grid[grid==0] = 253
                grid = buffer[(MIDY - sy) % R:SIZEY:R, (MIDX - sx + i) % R:SIZEX:R];  grid[grid==0] = 253

    def draw_params(self, buffer, markers=[]):
        if self.is_draw_params and 'params' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1]:
            s = 12
            b = self.world.params['b'].copy()
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
            pm = int(np.floor(self.world.params['m'] / 0.01))
            ps = int(np.floor(self.world.params['s'] / 0.002))
            if 0<=pm<=50 and 0<=ps<=60:
                buffer[51-pm+my, ps+mx] = 255

    def draw_arrow(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.params[k] for k in ('R', 'T')]
        midpoint = np.asarray(MID)
        dd = np.asarray([2]*DIM)
        if 'arrow' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1] and R > 2 and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
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
                    draw.line([p1[0], p1[1], p2[0], p2[1]], fill=254, width=1)
                    for (m,c) in [(m0,254),(m1,255),(m2,255),(m3,255)]:
                        p1 = (m+adj) * PIXEL - dd
                        p2 = (m+adj) * PIXEL + dd
                        draw.ellipse([p1[0], p1[1], p2[0], p2[1]], fill=c)
        del draw

    def draw_symmetry_title(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.params[k] for k in ('R', 'T')]
        if self.analyzer.is_calc_symmetry and 'scale' in markers and self.markers_mode in [0,1,4,5] and R > 2:
            k = self.analyzer.symm_sides
            self.draw_title(draw, 0, 'symmetry: {0} ({1})'.format(k, self.POLYGON_NAME[k] if k <= 10 else self.POLYGON_NAME[0]))
        del draw

    def draw_symmetry(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.params[k] for k in ('R', 'T')]
        midpoint = np.asarray([MIDX, MIDY])
        dd = np.asarray([1, 1]) * 2
        if self.analyzer.is_calc_symmetry and 'arrow' in markers and self.markers_mode in [0,1,2,3] and R > 2 and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
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
                                draw.line(tuple(m1*PIXEL) + tuple((m1-d1)*PIXEL), fill=254, width=1)
                            elif self.polar_mode in [2]:
                                #vertical
                                x = SIZETH * ((i / k - a/2/np.pi + 0.5) % 1)
                                draw.line((x*PIXEL, 0*PIXEL, x*PIXEL, SIZEY*PIXEL), fill=254, width=1)
                elif self.polar_mode in [4]:
                    #draw vertical lines per 5 sides in symmetry
                    for i in range(1, SIZEF, 5):
                        draw.line((i*2*PIXEL, 0*PIXEL, i*2*PIXEL, SIZEY*PIXEL), fill=254, width=1)
                        x0, y0 = i*2*PIXEL+2, MIDY*PIXEL
                        draw.text((x0,y0), str(i), fill=255, font=self.font)
                if self.polar_mode in [2,3,4]:
                    #draw horizontal line separate upper and lower panels
                    draw.line((0*PIXEL, SIZER*PIXEL, SIZEX*PIXEL, SIZER*PIXEL), fill=254, width=1)
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
                                # draw.line(tuple((m1-d1)*PIXEL) + tuple((m1-d2)*PIXEL), fill=c, width=d)
                                th1 = 270 - np.degrees(angle)
                                th2 = th1 - np.degrees(ww[r])
                                if th1 > th2: th1,th2 = th2,th1
                                draw.arc(tuple((m1-SIZER+r)*PIXEL) + tuple((m1+SIZER-r)*PIXEL), th1, th2, fill=c, width=1)
                                draw.ellipse(tuple((m1-d1)*PIXEL-dd) + tuple((m1-d1)*PIXEL+dd), fill=c)
                        elif self.polar_mode in [2]:
                            #horizontal
                            c = 255 if kk[r] == k else 254
                            for i in range(kk[r]):
                                x = SIZETH * ((i / kk[r] - aa[r]/2/np.pi + 0.5) % 1)
                                draw.line((x*PIXEL, r*PIXEL, (x-ww[r]/2/np.pi*SIZETH)*PIXEL, r*PIXEL), fill=c, width=1)
                                draw.ellipse((x*PIXEL-2, r*PIXEL-2, x*PIXEL+2, r*PIXEL+2), fill=c)
                        elif self.polar_mode in [4]:
                            #horizontal
                            c = 255
                            x = (kk[r]+1) * PIXEL//2
                            draw.line((x*PIXEL, r*PIXEL, (x-ww[r]/2/np.pi*SIZETH)*PIXEL, r*PIXEL), fill=c, width=1)
                            draw.ellipse((x*PIXEL-2, r*PIXEL-2, x*PIXEL+2, r*PIXEL+2), fill=c)
        del draw

    def draw_legend(self, markers=[]):
        draw = PIL.ImageDraw.Draw(self.img)
        R, T = [self.world.params[k] for k in ('R', 'T')]
        midpoint = np.asarray([MIDX, MIDY])
        dd = np.asarray([1, 1]) * 2
        if 'arrow' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1]:
            #draw speed legend
            x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-35
            draw.line([(x0-90,y0),(x0,y0)], fill=254, width=1)
            for (m,c) in [(0,254),(-10,255),(-50,255),(-90,255)]:
                draw.ellipse(tuple((x0+m,y0)-dd) + tuple((x0+m,y0)+dd), fill=c)
            draw.text((x0-95,y0-20), '2s', fill=255, font=self.font)
            draw.text((x0-55,y0-20), '1s', fill=255, font=self.font)
        if 'scale' in markers and self.markers_mode in [0,1,4,5] and self.polar_mode in [0,1]:
            #draw scale bar
            x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-20
            draw.text((x0+10,y0), '1mm', fill=255, font=self.font)
            draw.rectangle([(x0-R*PIXEL,y0+3),(x0,y0+7)], fill=255)
        if 'colormap' in markers and self.markers_mode in [0,2,4,6] and self.polar_mode in [0,1]:
            #draw colormap
            x0, y0 = SIZEX*PIXEL-20, SIZEY*PIXEL-70
            x1, y1 = SIZEX*PIXEL-15, 20
            dy = (y1-y0)/253
            draw.rectangle([(x0-1,y0+1),(x1+1,y1-1)], outline=254)
            for c in range(253):
                draw.rectangle([(x0,y0+dy*c),(x1,y0+dy*(c+1))], fill=c)
            draw.text((x0-25,y0-5), '0.0', fill=255, font=self.font)
            draw.text((x0-25,(y1+y0)//2-5), '0.5', fill=255, font=self.font)
            draw.text((x0-25,y1-5), '1.0', fill=255, font=self.font)
        del draw

    def cube_xy(self, d1, d2, d3, s):
        return (s + d1 - d3), (2*s + d1 - 2*d2 + d3)

    def draw_stats(self, is_current_series=True, is_small=True):
        draw = PIL.ImageDraw.Draw(self.img)
        series = self.analyzer.series
        current = self.analyzer.current
        name_x = self.analyzer.STAT_HEADERS[self.stats_x]
        name_y = self.analyzer.STAT_HEADERS[self.stats_y]
        if series != [] and is_current_series:
            series = [series[-1]]
        if series != [] and series != [[]]:
            X = [np.asarray([val[self.stats_x] for val in seg]) for seg in series if len(seg)>0]
            Y = [np.asarray([val[self.stats_y] for val in seg]) for seg in series if len(seg)>0]
            S = [seg[0][1] for seg in series if len(seg)>0]
            M = [seg[0][0] for seg in series if len(seg)>0]
            if name_x in ['n', 't']: X = [seg - seg.min() for seg in X]
            if name_y in ['n', 't']: Y = [seg - seg.min() for seg in Y]
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
            title_st_x = "X: {} ({:.2f}-{:.2f}) {:.2f}".format(name_x, xmin, xmax, current[self.stats_x])
            title_st_y = "Y: {} ({:.2f}-{:.2f}) {:.2f}".format(name_y, ymin, ymax, current[self.stats_y])
            if is_small:
                xmax = (xmax - xmin) * 4 + xmin
                ymax = (ymax - ymin) * 4 + ymin
                title_x, title_y = 5, (SIZEY * 3 // 4)*PIXEL - 32
            else:
                title_x, title_y = 5, 5
            draw.text((title_x,title_y), title_st_x, fill=255, font=self.font)
            draw.text((title_x,title_y+12), title_st_y, fill=255, font=self.font)
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
                    draw.rectangle([(x0,y0),(x1,y1)], outline=c, fill=None if is_valid else c)
                else:
                    is_square = name_x in ['x', 'y'] or name_x in ['x', 'y']
                    x = self.normalize(x, xmin, xmax, is_square, ymin, ymax) * (SIZEX*PIXEL - 10) + 5
                    y = (1-self.normalize(y, ymin, ymax, is_square, xmin, xmax)) * (SIZEY*PIXEL - 10) + 5
                if is_valid:
                    draw.line(list(zip(x, y)), fill=c, width=1)
        del draw

    def draw_psd(self, is_welch=True):
        draw = PIL.ImageDraw.Draw(self.img)
        T = self.world.params['T']
        series = self.analyzer.series
        name_x = self.analyzer.STAT_HEADERS[self.stats_x]
        name_y = self.analyzer.STAT_HEADERS[self.stats_y]
        if self.analyzer.is_calc_psd and self.analyzer.psd_freq is not None:
            self.draw_title(draw, 1, 'periodogram (Welch)' if is_welch else 'periodogram')
            freq = self.analyzer.psd_freq
            xmin, xmax = freq.min(), freq.max()
            self.analyzer.period = 1 / freq[np.argmax(self.analyzer.psd1)]
            self.analyzer.period_gen = self.analyzer.period * T
            for (n, psd, name) in zip([0,1], [self.analyzer.psd2, self.analyzer.psd1], [name_y, name_x]):
                if psd is not None and psd.shape[0] > 0:
                    #if len(psd.shape) > 1: psd = psd.max(axis=1)
                    c = 254 if n==0 else 255
                    ymin, ymax = psd.min(), psd.max()
                    period = 1 / freq[np.argmax(psd)]
                    x = self.normalize(freq, xmin, xmax) * (SIZEX*PIXEL - 10) + 5
                    y = (1-self.normalize(psd, ymin, ymax)) * (SIZEY*PIXEL - 10) + 5
                    draw.line(list(zip(x, y)), fill=c, width=1)
                    self.draw_title(draw, 3-n, 'period from {0} = {1:.2f}s'.format(name, period), color=c)
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
        b = self.world.params['b'].copy()
        B = len(b)
        if B > 1 and i < B:
            b[i] = min(1, max(0, b[i] + Fraction(d, s)))
            #check if a least one fraction = 1
            # if b.count(1) >= 1:
            self.world.params['b'] = b
            self.automaton.calc_kernel()
            self.check_auto_load()

    def adjust_b(self, d):
        B = len(self.world.params['b'])
        if B <= 0:
            self.world.params['b'] = [1]
        elif B >= 5:
            self.world.params['b'] = self.world.params['b'][0:5]
        else:
            self.world.params['R'] = self.world.params['R'] * B // (B-d)
            # temp_R = self.tx['R']
            # self.tx['R'] = self.tx['R'] * (B-d) // B
            # self.transform_world()
            # self.world.params['R'] = temp_R
            # self.automaton.calc_kernel()

    def _recur_write_csv(self, dim, writer, cells):
        if dim < DIM-2:
            writer.writerow(['<{0}D>'.format(DIM-dim)])
            for e in cells:
                self._recur_write_csv(dim+1, writer, e)
            writer.writerow(['</{0}D>'.format(DIM-dim)])
        else:
            writer.writerows(cells)
            writer.writerow([])

    def copy_world(self, type='JSON'):
        A = copy.deepcopy(self.world)
        A.crop()
        if type == 'JSON':
            data = A.to_data()
            self.clipboard_st = json.dumps(data, separators=(',', ':'), ensure_ascii=False) + ','
        elif type == 'CSV':
            stio = io.StringIO()
            writer = csv.writer(stio, delimiter=',', lineterminator='\n')
            self._recur_write_csv(0, writer, A.cells)
            self.clipboard_st = stio.getvalue()

        self.window.clipboard_clear()
        self.window.clipboard_append(self.clipboard_st)
        STATUS.append("> copied board to clipboard as "+type)

    def paste_world(self):
        try:
            st = self.clipboard_st = self.window.clipboard_get()
            if 'cells' in st:
                data = json.loads(st.rstrip(', \t\r\n'))
                self.load_part(self.world, Board.from_data(data))
                self.info_type = 'params'
            elif '\t' in st or ',' in st:
                delim = '\t' if '\t' in st else ','
                stio = io.StringIO(st)
                #stio.setvalue(st)
                reader = csv.reader(stio, delimiter=delim, lineterminator='\n')
                cells = np.asarray([[float(c) if c != '' else 0 for c in row] for row in reader])
                self.load_part(self.world, Board.from_values(cells))
            else:
                self.load_animal_code(self.world, st)
                self.info_type = 'animal'
        except (tk.TclError, ValueError, json.JSONDecodeError) as e:
            STATUS.append("> no valid JSON or CSV in clipboard")

    def save_world(self, is_seq=False):
        A = copy.deepcopy(self.world)
        A.crop()
        data = A.to_data()
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
                file.write('x = '+str(A.cells.shape[0])+', y = '+str(A.cells.shape[1])+', rule = Lenia('+A.params2st()+')\n')
                file.write(data['cells'].replace('$','$\n')+'\n')
            data['cells'] = [row if row.endswith('!') else row+'%' for row in data['cells'].split('%')]
            with open(path+'.json', 'w', encoding='utf-8')  as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            with open(path+'.csv', 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow([self.analyzer.stat_name(x=x) for x in self.analyzer.STAT_HEADERS])
                writer.writerows([e for l in self.analyzer.series for e in l])
            STATUS.append("> data and image saved to '"+path+".*'")
            self.is_save_image = True
        except IOError as e:
            STATUS.append("I/O error({0}): {1}".format(e.errno, e.strerror))

    def backup_world(self):
        self.backup_world = copy.deepcopy(self.world)
        self.backup_automaton = copy.deepcopy(self.automaton)
        self.backup_analyzer = copy.deepcopy(self.analyzer)
        STATUS.append("> time saved")

    def restore_world(self):
        if self.backup_world is not None:
            self.world = copy.deepcopy(self.backup_world)
            self.automaton = copy.deepcopy(self.backup_automaton)
            self.analyzer = copy.deepcopy(self.backup_analyzer)
            self.automaton.world = self.world
            self.analyzer.world = self.world
            self.analyzer.automaton = self.automaton
            self.world_updated()
            STATUS.append("> time loaded")

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

    SHIFT_KEYS = {'asciitilde':'quoteleft', 'exclam':'1', 'at':'2', 'numbersign':'3', 'dollar':'4', 'percent':'5', 'asciicircum':'6', 'ampersand':'7', 'asterisk':'8', 'parenleft':'9', 'parenright':'0', 'underscore':'-', 'plus':'equal', \
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
            's+1':'3GH2n', 's+2':'3GG2g', 's+3':'K5s', 's+4':'K7a', 's+5':'K9a', 's+6':'3R5t:2', 's+7':'3R6s:4', 's+8':'2D10rb', 's+9':'4F14x9xR5r', 's+0':'~ggun', \
            'c+1':'4Q5m:2', 'c+2':'3ECv', 'c+3':'2L4m:2', 'c+4':'3R6n', 'c+5':'4D10v:2', 'c+6':'', 'c+7':'', 'c+8':'', 'c+9':'', 'c+0':'bbug',
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
        inc_or_dec     = 1 if 's+' not in k else -1
        inc_10_or_1    = 10 if 's+' not in k else 1
        rot_15_or_1    = 15 if 's+' not in k else 1
        inc_1_or_10    = 1 if 's+' not in k else 10
        inc_mul_or_not = 1 if 's+' not in k else 0
        double_or_not  = 2 if 's+' not in k else 1
        inc_or_not     = 0 if 's+' not in k else 1

        #inc_10_or_1   = (10 if 's+' not in k else 1) if 'c+' not in k else 0 
        #inc_big_or_not = 0 if 'c+' not in k else 1

        is_ignore = False
        if not self.is_internal_key and k not in ['backspace', 'delete']:
            self.stop_search()

        #run [esc enter space c+g]
        if k in ['escape']: self.is_closing = True; self.close()
        elif k in ['enter', 'return']: self.is_run = not self.is_run; self.run_counter = -1; self.info_type = 'info'
        elif k in [' ', 'space']: self.is_run = True; self.run_counter = 1; self.info_type = 'info'
        elif k in ['c+space']: self.is_run = True; self.run_counter = self.samp_freq; self.info_type = 'info'
        elif k in ['c+g']:
            if self.automaton.has_gpu:
                self.automaton.is_gpu = not self.automaton.is_gpu
        elif k in ['c+quoteleft']: self.is_advanced_menu = not self.is_advanced_menu; self.create_menu()
        #sampling,symmetry [[ ] \]
        elif k in ['bracketright', 's+bracketright']: self.samp_freq = self.samp_freq + (9 if self.samp_freq==1 and inc_10_or_1==10 else inc_10_or_1); self.info_type = 'info'
        elif k in ['bracketleft',  's+bracketleft']:  self.samp_freq = self.samp_freq - inc_10_or_1; self.info_type = 'info'
        elif k in ['a+bracketright']: self.samp_freq = int(round((round(self.samp_freq / self.analyzer.period_gen) + 1) * self.analyzer.period_gen)); self.info_type = 'info'
        elif k in ['a+bracketleft']: self.samp_freq = max(1, int(round((round(self.samp_freq / self.analyzer.period_gen) - 1) * self.analyzer.period_gen))); self.info_type = 'info'
        elif k in ['c+bracketright']: self.samp_sides += 1; self.info_type = 'angular'
        elif k in ['c+bracketleft']: self.samp_sides -= 1; self.info_type = 'angular'
        elif k in ['backslash']: self.toggle_auto_rotate_from_sampling()
        elif k in ['s+backslash']: self.samp_freq = 1; self.info_type = 'info'
        elif k in ['c+backslash']: self.is_samp_clockwise = not self.is_samp_clockwise; self.info_type = 'angular'
        #display [< > tab `]
        elif k in ['s+period']: self.colormap_id = (self.colormap_id + 1) % len(self.colormaps)
        elif k in ['s+comma']: self.colormap_id = (self.colormap_id - 1) % len(self.colormaps)
        elif k in ['tab', 's+tab']: self.show_what = (self.show_what + inc_or_dec) % 4
        elif k in ['c+tab', 's+c+tab']: self.show_what = (self.show_what + inc_or_dec) % 8
        elif k in ['quoteleft', 's+quoteleft']: self.polar_mode = (self.polar_mode + inc_or_dec) % 5 if DIM==2 else 0
        #params [qa ws ed rf tg]
        elif k in ['q', 's+q']: self.world.params['m'] += inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['a', 's+a']: self.world.params['m'] -= inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['w', 's+w']: self.world.params['s'] += inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['s', 's+s']: self.world.params['s'] -= inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['t', 's+t']: self.world.params['T'] = max(1, self.world.params['T'] *  double_or_not + inc_or_not); self.analyzer.new_segment(); self.info_type = 'params'
        elif k in ['g', 's+g']: self.world.params['T'] = max(1, self.world.params['T'] // double_or_not - inc_or_not); self.analyzer.new_segment(); self.info_type = 'params'
        elif k in ['r', 's+r']: self.tx['R'] = max(1, self.tx['R'] + inc_10_or_1); self.transform_world(); self.info_type = 'params'
        elif k in ['f', 's+f']: self.tx['R'] = max(1, self.tx['R'] - inc_10_or_1); self.transform_world(); self.info_type = 'params'
        elif k in ['e', 's+e']: self.world.param_P = max(0, self.world.param_P + inc_10_or_1); self.info_type = 'info'
        elif k in ['d', 's+d']: self.world.param_P = max(0, self.world.param_P - inc_10_or_1); self.info_type = 'info'
        elif k in ['c+d']: self.world.param_P = 0; self.info_type = 'info'
        elif k in ['c+r']: self.tx['R'] = DEF_R; self.transform_world(); self.info_type = 'params'
        elif k in ['c+f']: self.tx['R'] = self.fore.params['R'] if self.fore else DEF_R; self.transform_world(); self.info_type = 'params'
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
        elif k in ['semicolon']:   self.world.params['b'].append(0); self.adjust_b(+1); self.info_type = 'params'
        elif k in ['s+semicolon']: self.world.params['b'].pop();     self.adjust_b(-1); self.info_type = 'params'
        #search [c+qa]
        elif k in ['c+q', 's+c+q']: self.toggle_search(+1, 's+' in k)
        elif k in ['c+a', 's+c+a']: self.toggle_search(-1, 's+' in k)
        #options [c+yuiop]
        elif k in ['c+y', 's+c+y']: self.world.params['kn'] = (self.world.params['kn'] + inc_or_dec - 1) % len(self.automaton.kernel_core) + 1; self.automaton.calc_kernel(); self.info_type = 'kn'
        elif k in ['c+u', 's+c+u']: self.world.params['gn'] = (self.world.params['gn'] + inc_or_dec - 1) % len(self.automaton.growth_func) + 1; self.info_type = 'gn'
        elif k in ['c+i']: self.automaton.is_soft_clip = not self.automaton.is_soft_clip
        elif k in ['c+o']: self.automaton.is_multi_step = not self.automaton.is_multi_step
        elif k in ['c+p']: self.automaton.is_inverted = not self.automaton.is_inverted; self.world.params['T'] *= -1; self.world.params['m'] = 1 - self.world.params['m']; self.world.cells = 1 - self.world.cells
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
        elif k in ['equal']: self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['s+equal']: self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
        elif k in ['c+equal']: self.tx['flip'] = 2 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['s+c+equal']: self.tx['flip'] = 3 if self.tx['flip'] != 0 else -1; self.transform_world()
        # elif k in ['equal']: self.tx['flip'] = 4 if self.tx['flip'] != 0 else -1; self.transform_world()
        #autocenter [']
        elif k in ['quoteright']: self.is_auto_center = not self.is_auto_center
        elif k in ['s+quoteright']: self.auto_rotate_mode = (self.auto_rotate_mode + 1) % 3 if DIM==2 else 0
        elif k in ['c+quoteright']: self.is_auto_center = True; self.auto_rotate_mode = 2; self.analyzer.is_calc_symmetry = True; self.stats_mode = 5; self.samp_freq = int(round(self.analyzer.period_gen*2))
        elif k in ['s+c+quoteright']: self.is_auto_center = False; self.auto_rotate_mode = 0; self.analyzer.is_calc_symmetry = False; self.stats_mode = 0; self.samp_freq = 1
        #animals [zxcv 1-0]
        elif k in ['z']: self.load_animal_id(self.world, self.animal_id); self.world_updated(); self.info_type = 'animal'
        elif k in ['c']: self.load_animal_id(self.world, self.animal_id - inc_1_or_10); self.world_updated(); self.info_type = 'animal'
        elif k in ['v']: self.load_animal_id(self.world, self.animal_id + inc_1_or_10); self.world_updated(); self.info_type = 'animal'
        elif k in ['s+c']: self.search_animal(self.world, 'family: ', -1); self.info_type = 'animal'
        elif k in ['s+v']: self.search_animal(self.world, 'family: ', +1); self.info_type = 'animal'
        elif k in ['s+z']: self.load_animal_id(self.world, self.animal_id); self.world.params['kn'] = self.world.params['gn'] = 2; self.automaton.calc_kernel(); self.info_type = 'animal'
        elif k in ['c+b']: self.load_found_animal_id(self.world, self.found_animal_id); self.world_updated(); self.info_type = 'params'
        elif k in ['s+b']: self.load_found_animal_id(self.world, self.found_animal_id - 1); self.world_updated(); self.info_type = 'params'
        elif k in ['b']: self.load_found_animal_id(self.world, self.found_animal_id + 1); self.world_updated(); self.info_type = 'params'
        elif k in ['x', 's+x']: self.load_part(self.world, self.fore, is_random=True, is_replace=False, repeat=inc_1_or_10); self.world_updated()
        elif k in ['c+z']: self.is_auto_load = not self.is_auto_load
        elif k in ['s+c+z']: self.read_animals(); self.read_found_animals(); self.create_menu()
        elif k in ['s+c+x']: self.is_layered = not self.is_layered
        elif k in [m+str(i) for i in range(10) for m in ['','s+','c+','s+c+','a+','s+a+']]: self.load_animal_code(self.world, self.ANIMAL_KEY_LIST.get(k)); self.world_updated(); self.info_type = 'animal'
        #random [del n m]
        elif k in ['backspace', 'delete']: self.clear_world()
        elif k in ['n', 's+n']: self.random_world(is_reseed='s+' in k)
        #elif k in ['c+n', 's+c+n']: self.cppn_world(is_reseed='s+' in k)
        elif k in ['m']: self.random_world_and_params(); self.info_type = 'params'
        elif k in ['c+m']: self.toggle_search(0); self.random_world_and_params(); self.info_type = 'params'
        #copy,paste,save,record [c+xcv c+s c+w]
        elif k in ['c+c']: self.copy_world(type='JSON')
        elif k in ['c+x']: self.copy_world(type='CSV')
        elif k in ['c+v']: self.paste_world()
        elif k in ['c+s', 's+c+s']: self.save_world(is_seq='s+' in k)
        elif k in ['s+c+c']: self.backup_world()
        elif k in ['s+c+v']: self.restore_world()
        elif k in ['c+w', 's+c+w']: self.recorder.toggle_recording(is_save_frames='s+' in k)
        #stats [hjkl]
        elif k in ['h', 's+h']: self.markers_mode = (self.markers_mode + inc_or_dec) % 8
        elif k in ['c+h']: self.is_show_fps = not self.is_show_fps
        elif k in ['j', 's+j']: self.stats_mode = (self.stats_mode + inc_or_dec) % 7; self.info_type = 'stats'
        elif k in ['k', 's+k']: self.stats_x = self.change_stat_axis(self.stats_x, self.stats_y, inc_or_dec); self.info_type = 'stats'
        elif k in ['l', 's+l']: self.stats_y = self.change_stat_axis(self.stats_y, self.stats_x, inc_or_dec); self.info_type = 'stats'
        elif k in ['c+j']: self.analyzer.clear_segment()
        elif k in ['a+j']: self.stats_mode = 5  # periodogram
        elif k in ['s+c+j']: self.analyzer.clear_series()
        elif k in ['c+k']: self.analyzer.is_trim_segment = not self.analyzer.is_trim_segment
        elif k in ['c+l']: self.is_group_params = not self.is_group_params
        elif k in ['s+c+k']: self.stats_mode = 1; self.stats_x = 4; self.stats_y = 5; self.analyzer.is_trim_segment = True; self.info_type = 'stats'
        elif k in ['s+c+l']: self.stats_mode = 1; self.stats_x = 11; self.stats_y = 12; self.analyzer.is_trim_segment = False; self.info_type = 'stats'
        #info [,./]
        elif k in ['comma']: self.info_type = 'animal'
        elif k in ['period']: self.info_type = 'params'
        elif k in ['slash']: self.info_type = 'info'
        elif k in ['s+slash']: self.info_type = 'angular'
        #misc
        # elif k in ['-', 'quoteright']
        # elif k in ['c+slash']: m = self.menu.children[self.menu_values['animal'][0]].children['!menu']; m.post(self.window.winfo_rootx(), self.window.winfo_rooty())
        elif k.endswith('_l') or k.endswith('_r'): is_ignore = True
        else: self.excess_key = k

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
        self.samp_rotate = (-1 if self.is_samp_clockwise else +1) * 360/self.samp_sides/self.samp_gen*self.world.params['T'] if self.auto_rotate_mode in [3] else 0
        if not is_ignore and self.is_loop:
            self.roundup(self.world.params)
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)
            self.update_menu()

        # if not is_ignore:
            # v1 = self.target.cells.reshape(-1, 1)
            # v2 = self.world.cells.reshape(-1, 1)
            # print('fitness: {:.4f}'.format(-np.linalg.norm(v1 - v2)))

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
        ctrl = 'Ctrl+' if (is_windows or acc in ['c+Space','c+Q','c+H']) else 'Command+'
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
                        new_list = ('{name} {cname}'.format(**data), [])
                        stack[-1].append(new_list)
                        stack.append(new_list[1])
            else:
                stack[-1].append('&{id}|{code} - {name} {cname}|'.format(id=id, **data))
            id += 1
        return root

    def get_nested_attr(self, name):
        obj = self
        for n in name.split('.'):
            obj = getattr(obj, n)
        return obj
    def get_value_text(self, name):
        if name=='animal': return '#'+str(self.animal_id+1)+' '+self.world.long_name()
        elif name=='kn': return ["Exponential","Polynomial","Step","Staircase"][self.world.params.get('kn') - 1]
        elif name=='gn': return ["Exponential","Polynomial","Step"][self.world.params.get('gn') - 1]
        elif name=='colormap_id': return ["Vivid blue/red (Jet)","Paul Tol's Rainbow","Vivid green/purple","Vivid red/green","Pale blue/red","Pale green/purple","Pale yellow/green","White/black","Black/white"][self.colormap_id]
        elif name=='show_what': return ["World","Potential","Field","Kernel","World FFT","Potential FFT","Gradient","Change"][self.show_what]
        elif name=='polar_mode': return ["Off","Symmetry","Polar","History","Strength"][self.polar_mode]
        elif name=='auto_rotate_mode': return ["Off","Arrow","Symmetry","Sampling"][self.auto_rotate_mode]
        elif name=='markers_mode': 
            st = []
            if self.markers_mode in [0,1,2,3]: st.append("Grid")
            if self.markers_mode in [0,1,4,5]: st.append("Ruler")
            if self.markers_mode in [0,2,4,6]: st.append("Colors")
            return ",".join(st) if st != [] else "None"
        elif name=='stats_mode': return ["None","Corner","Overlay","Segment","All segments","Periodogram","Recurrence plot"][self.stats_mode]
        elif name=='stats_x': return self.analyzer.stat_name(i=self.stats_x)
        elif name=='stats_y': return self.analyzer.stat_name(i=self.stats_y)
        elif name=='z_axis': return str(DIM-self.z_axis)
        elif name=='mask_rate': return '{}%'.format(self.automaton.mask_rate * 10)
        elif name=='add_noise': return '{}%'.format(self.automaton.add_noise * 10)
    def update_menu(self):
        for name in self.menu_vars:
            self.menu_vars[name].set(self.get_nested_attr(name))
        for (name, info) in self.menu_params.items():
            # value = '['+Board.fracs2st(self.world.params[name])+']' if name=='b' else self.world.params[name]
            # self.menu.children[info[0]].entryconfig(info[1], label='{text} ({param} = {value})'.format(text=info[2], param=name, value=value))
            value = self.get_nested_attr(name)
            self.menu.children[info[0]].entryconfig(info[1], label='{text} [{value}]'.format(text=info[2], value=value))
        for (name, info) in self.menu_values.items():
            value = self.get_value_text(name)
            self.menu.children[info[0]].entryconfig(info[1], label='{text} [{value}]'.format(text=info[2], value=value))

    PARAM_TEXT = {'m':'Field center', 's':'Field width', 'R':'Space units', 'T':'Time units', 'dr':'Space step', 'dt':'Time step', 'b':'Kernel peaks'}
    VALUE_TEXT = {'animal':'Lifeform', 'kn':'Kernel core', 'gn':'Field func', 'show_what':'Show', 'colormap_id':'Colors'}
    def create_menu(self):
        self.menu_vars = {}
        self.menu_params = {}
        self.menu_values = {}
        self.menu = tk.Menu(self.window, tearoff=True)
        self.window.config(menu=self.menu)

        self.menu.add_cascade(label='Lenia', menu=self.create_submenu(self.menu, [
            '^is_run|Running|Return', '|Once|Space'] + 
            (['^automaton.is_gpu|Use GPU|c+G'] if self.automaton.has_gpu else ['|No GPU available|']) + [None,
            '@show_what|Display|Tab', '@colormap_id|Colors|s+Period|>', None,
            '|*Show lifeform name|Comma|,', '|*Show params|Period|.', '|*Show info|Slash|/', '|*Show auto-rotate info|s+Slash|?', None,
            '|Save data & image|c+S', '|*Save next in sequence|s+c+S', 
            '^recorder.is_recording|Record video & gif|c+W', '|*Record with frames saved|s+c+W', None,
            '^is_advanced_menu|Advanced menu|c+QuoteLeft|c+`', None,
            '|Quit|Escape']))

        self.menu.add_cascade(label='Edit', menu=self.create_submenu(self.menu, [
            '|Clear|Backspace', '|Random|N', '|*Random (last seed)|s+N', '|Random cells & params|M', None,
            '|Flip vertically|Equal|=', '|Flip horizontally|s+Equal|+',
            '|Mirror horizontally|c+Equal|c+=', '|Mirror flip|s+c+Equal|c++', None,
            '|Copy|c+C', '|*Copy as CSV|c+X', '|Paste|c+V', None,
            '^is_auto_load|*Auto put mode (place/paste/random)|c+Z', '^is_layered|*Layer mode|s+c+X']))

        # for (key, code) in self.ANIMAL_KEY_LIST.items():
            # id = self.get_animal_id(code)
            # if id: items2.append('|{name} {cname}|{key}'.format(**self.animal_data[id], key=key))
        self.menu.add_cascade(label='Lifeform', menu=self.create_submenu(self.menu, [
            '|Place at center|Z', '|Add at random|X',
            '|Previous|C', '|Next|V', '|Previous family|s+C', '|Next family|s+V', None,
            '|*Start auto find (any key to stop)|c+M', '|*Place found|c+B', '|*Previous found|s+B', '|*Next found|B', None,
            '|Shortcuts 1-10|1', '|Shortcuts 11-20|s+1', '|Shortcuts 21-30|c+1', None,
            '|*Reload list|s+c+Z']))

        self.menu.add_cascade(label='List', menu=self.create_submenu(self.menu, 
            self.get_animal_nested_list()))

        self.menu.add_cascade(label='Space', menu=self.create_submenu(self.menu, [
            '^is_auto_center|Auto-center mode|QuoteRight|\'', None,
            '|(Small adjust)||s+Up', #'|(Large adjust)||m+Up',
            '|Move up|Up', '|Move down|Down', '|Move left|Left', '|Move right|Right'] + 
            (['|Move front|PageUp', '|Move back|PageDown', None,
            '|*(Small adjust)||s+Home', '|*Slice front|Home', '|*Slice back|End', '|*Reset slice|c+Home', 
            '^is_show_slice|*Show Z slice|c+End', '@z_axis|*Change Z axis|s+c+Home'] 
            if DIM > 2 else []) ))

        self.menu.add_cascade(label='Polar', menu=self.create_submenu(self.menu,
            (['@polar_mode|Polar mode|QuoteLeft|`', '@auto_rotate_mode|*Auto-rotate by|s+QuoteRight|"', None,
            '|(Small adjust)||s+c+Up', '|Rotate anti-clockwise|c+Up', '|Rotate clockwise|c+Down'] 
            if DIM == 2 else [
            '|(Small adjust)||s+c+Up', '|Rotate right|c+Right', '|Rotate left|c+Left', 
            '|Rotate up|c+Up', '|Rotate down|c+Down',
            '|Rotate anti-clockwise|c+PageUp', '|Rotate clockwise|c+PageDown']) + [None,
            '|*(Small adjust)||s+]', '|*Sampling period + 10|BracketRight|]', '|*Sampling period - 10|BracketLeft|[', 
            '|*Clear sampling|s+BackSlash|bar', '|*Run one sampling period|c+Space', None] +
            (['|*Auto-rotate by sampling|BackSlash|\\', '|*Symmetry axes + 1|c+BracketRight|c+]', 
            '|*Symmetry axes - 1|c+BracketLeft|c+[', '^is_samp_clockwise|*Clockwise|c+BackSlash|c+\\'] 
            if DIM == 2 else []) ))

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
            '|*Reset space|c+R', '|*Lifeform\'s original size|c+F', None,
            '|Slower time (T * 2)|T', '|Faster time (T / 2)|G', None,
            ('Peaks', items2)]))

        self.menu.add_cascade(label='Options', menu=self.create_submenu(self.menu, [
            '|*Search growth higher|c+Q', '|*Search growth lower|c+A', None,
            '@kn|Kernel core|c+Y', '@gn|Growth mapping|c+U', None,
            '^automaton.is_soft_clip|*Use soft clip|c+I', '^automaton.is_multi_step|*Use multi-step|c+O', 
            '^automaton.is_inverted|*Invert mode|c+P', None,
            '@mask_rate|*Async rate|s+c+I', '@add_noise|*Noise rate|s+c+O', 
            '|*Reset async & noise|s+c+P']))

        self.menu.add_cascade(label='Stats', menu=self.create_submenu(self.menu, [
            '@markers_mode|Show marks|H', '^is_show_fps|*Show FPS|c+H', None,
            '@stats_mode|Show stats|J', '@stats_x|Stats X axis|K', '@stats_y|Stats Y axis|L', None,
            '|*Clear segment|c+J', '|*Clear all segments|s+c+J', 
            '^analyzer.is_trim_segment|*Trim segments|c+K', '^is_group_params|*Group by params|c+L']))

    def get_info_st(self):
        P = str(self.world.param_P)
        if P == '0': P = '∞'
        return 'gen={}, t={}s, dt={}s, P={}, world={}, pixel={}, sampl={}'.format(self.automaton.gen, self.automaton.time, 1/self.world.params['T'], P, 'x'.join(str(size) for size in SIZE), PIXEL, self.samp_freq)

    def get_angular_st(self):
        if self.auto_rotate_mode in [3]:
            return 'auto-rotate: {} axes={} sampl={} speed={:.2f}'.format('clockwise' if self.is_samp_clockwise else 'anti-clockwise', self.samp_sides, self.samp_gen, self.samp_rotate)
        else:
            return 'not in auto-rotate mode'

    def update_info_bar(self):
        global STATUS
        if self.excess_key:
            #print(self.excess_key)
            self.excess_key = None
        if self.info_type or STATUS or self.is_show_fps:
            info_st = ""
            if STATUS: info_st = "\n".join(STATUS)
            elif self.is_show_fps and self.fps: info_st = 'FPS: {0:.1f}'.format(self.fps)
            elif self.info_type == 'params': info_st = self.world.params2st(); self.is_draw_params = True
            elif self.info_type == 'animal': info_st = self.world.long_name(); self.is_draw_params = True
            elif self.info_type == 'info': info_st = self.get_info_st()
            elif self.info_type == 'angular': info_st = self.get_angular_st()
            elif self.info_type == 'stats': info_st = 'X axis: {0}, Y axis: {1}'.format(self.analyzer.stat_name(i=self.stats_x), self.analyzer.stat_name(i=self.stats_y))
            elif self.info_type == 'slice': info_st = 'slice: {0}, Z axis: {1}th dim'.format(self.z_slices, DIM-self.z_axis)
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
                self.automaton.calc_once()
                self.analyzer.center_world()
                self.analyzer.calc_stats(self.show_what, psd_x=self.stats_x, psd_y=self.stats_y, is_welch=True)
                self.analyzer.add_stats(psd_y=self.stats_y)
                if not self.is_layered:
                    self.back = None
                    self.clear_transform()
                if self.search_mode is not None:
                    self.do_search()
                if self.run_counter != -1:
                    self.run_counter -= 1
                    if self.run_counter == 0:
                        self.is_run = False
            is_show_gen = self.automaton.gen % 1000 == 0 and self.automaton.gen // 1000 > 3
            if (self.search_mode != 0 and counter % self.samp_freq == 0) or \
               (self.search_mode == 0 and self.is_show_search):
                self.update_window()
                if not is_show_gen: self.update_info_bar()
                self.is_show_search = False
            if is_show_gen:
                self.info_type = 'info'
                self.update_info_bar()

    def print_help(self):
        print('''Lenia in n-Dimensions    by Bert Chan 2020    Run '{0} -h' for startup arguments.'''.format(sys.argv[0]))

if __name__ == '__main__':
    lenia = Lenia()
    #lenia.print_help()
    if lenia.ANIMAL_KEY_LIST != {} and lenia.ANIMAL_KEY_LIST['1'] != '':
        lenia.load_animal_code(lenia.world, lenia.ANIMAL_KEY_LIST['1'])
    else:
        lenia.world.params = {'R':DEF_R, 'T':10, 'b':[1], 'm':0.14, 's':0.014, 'kn':1, 'gn':1}
        lenia.automaton.calc_kernel()
        lenia.random_world()
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
