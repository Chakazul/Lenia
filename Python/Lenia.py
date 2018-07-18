import numpy as np                          # pip3 install numpy
import scipy                                # pip3 install scipy
import scipy.ndimage as snd
import reikna.fft, reikna.cluda             # pip3 install pyopencl/pycuda, reikna
from PIL import Image, ImageTk, ImageDraw   # pip3 install pillow
try: import tkinter as tk
except: import Tkinter as tk
from fractions import Fraction
import copy, re, itertools, json, csv
import os, sys, subprocess, datetime, time
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from snd.zoom()

P2, PIXEL_BORDER = 0,0    # 4,2  3,1  2,1  0,0
X2, Y2 = 9,9    # 10,9  9,8  8,8  1<<9=512
PIXEL = 1 << P2; SIZEX, SIZEY = 1 << (X2-P2), 1 << (Y2-P2)
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1280//PIXEL, 720//PIXEL    # 720p HD
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1920//PIXEL, 1080//PIXEL    # 1080p HD
MIDX, MIDY = int(SIZEX / 2), int(SIZEY / 2)
DEF_R = max(min(SIZEX, SIZEY) // 4 //5*5, 13)
EPSILON = 1e-10
ROUND = 10
FPS_FREQ = 20
STATUS = []
is_windows = (os.name == 'nt')

class Board:
	def __init__(self, size=[0,0]):
		self.names = ['', '', '']
		self.params = {'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
		self.cells = np.zeros(size)

	@classmethod
	def from_values(cls, names, params, cells):
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
	def arr2rle(A, is_shorten=True):
		''' RLE = Run-length encoding: 
			http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
			http://golly.sourceforge.net/Help/formats.html#rle
			https://www.rosettacode.org/wiki/Run-length_encoding#Python
			0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO '''
		V = np.rint(A*255).astype(int).tolist()  # [[255 255] [255 0]]
		code_arr = [ [' .' if v==0 else ' '+chr(ord('A')+v-1) if v<25 else chr(ord('p')+(v-25)//24) + chr(ord('A')+(v-25)%24) for v in row] for row in V]  # [[yO yO] [yO .]]
		if is_shorten:
			rle_groups = [ [(len(list(g)),c.strip()) for c,g in itertools.groupby(row)] for row in code_arr]  # [[(2 yO)] [(1 yO) (1 .)]]
			for row in rle_groups:
				if row[-1][1]=='.': row.pop()  # [[(2 yO)] [(1 yO)]]
			st = '$'.join(''.join([(str(n) if n>1 else '')+c for n,c in row]) for row in rle_groups) + '!'  # "2 yO $ 1 yO"
		else:
			st = '$'.join(''.join(row) for row in code_arr) + '!'
		# print(sum(sum(r) for r in V))
		return st

	@staticmethod
	def rle2arr(st):
		rle_groups = re.findall('(\d*)([p-y]?[.boA-X$])', st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
		code_list = sum([[c] * (1 if n=='' else int(n)) for n,c in rle_groups], [])  # [yO yO $ yO]
		code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
		V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row if c!='' ] for row in code_arr]  # [[255 255] [255]]
		# lines = st.rstrip('!').split('$')
		# rle = [re.findall('(\d*)([p-y]?[.boA-X])', row) for row in lines]
		# code = [ sum([[c] * (1 if n=='' else int(n)) for n,c in row], []) for row in rle]
		# V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row ] for row in code]
		maxlen = len(max(V, key=len))
		A = np.array([row + [0] * (maxlen - len(row)) for row in V])/255  # [[1 1] [1 0]]
		# print(sum(sum(r) for r in V))
		return A

	@staticmethod
	def fracs2st(B):
		return ','.join([str(f) for f in B])

	@staticmethod
	def st2fracs(st):
		return [Fraction(st) for st in st.split(',')]

	def clear(self):
		self.cells.fill(0)

	def add(self, part, shift=[0,0]):
		# assert self.params['R'] == part.params['R']
		h1, w1 = self.cells.shape
		h2, w2 = part.cells.shape
		h, w = min(h1, h2), min(w1, w2)
		i1, j1 = (w1 - w)//2 + shift[1], (h1 - h)//2 + shift[0]
		i2, j2 = (w2 - w)//2, (h2 - h)//2
		# self.cells[j:j+h, i:i+w] = part.cells[0:h, 0:w]
		vmin = np.amin(part.cells)
		for y in range(h):
			for x in range(w):
				if part.cells[j2+y, i2+x] > vmin:
					self.cells[(j1+y)%h1, (i1+x)%w1] = part.cells[j2+y, i2+x]
		return self

	def transform(self, tx, mode='RZSF', is_world=False):
		if 'R' in mode and tx['rotate'] != 0:
			self.cells = snd.rotate(self.cells, tx['rotate'], reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')
		if 'Z' in mode and tx['R'] != self.params['R']:
			# print('* {} / {}'.format(tx['R'], self.params['R']))
			shape_orig = self.cells.shape
			self.cells = snd.zoom(self.cells, tx['R'] / self.params['R'], order=0)
			if is_world:
				self.cells = Board(shape_orig).add(self).cells
			self.params['R'] = tx['R']
		if 'F' in mode and tx['flip'] != -1:
			if tx['flip'] in [0,1]: self.cells = np.flip(self.cells, axis=tx['flip'])
			elif tx['flip'] == 2: self.cells[:, :-MIDX-1:-1] = self.cells[:, :MIDX]
			elif tx['flip'] == 3: self.cells[:, :-MIDX-1:-1] = self.cells[::-1, :MIDX]
		if 'S' in mode and tx['shift'] != [0, 0]:
			self.cells = snd.shift(self.cells, tx['shift'], order=0, mode='wrap')
			# self.cells = np.roll(self.cells, tx['shift'], (1, 0))
		return self

	def add_transformed(self, part, tx):
		part = copy.deepcopy(part)
		self.add(part.transform(tx, mode='RZF'), tx['shift'])
		return self

	def crop(self):
		vmin = np.amin(self.cells)
		coords = np.argwhere(self.cells > vmin)
		y0, x0 = coords.min(axis=0)
		y1, x1 = coords.max(axis=0) + 1
		self.cells = self.cells[y0:y1, x0:x1]
		return self

class Automaton:
	kernel_core = {
		0: lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
		1: lambda r: np.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
		2: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
		3: lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
	}
	field_func = {
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
		self.X = None
		self.Y = None
		self.D = None
		self.gen = 0
		self.time = 0
		self.is_multi_step = False
		self.is_soft_clip = False
		self.is_inverted = False
		self.kn = 1
		self.gn = 1
		self.is_gpu = True
		self.has_gpu = True
		self.compile_gpu(self.world.cells)
		self.calc_kernel()

	def kernel_shell(self, r):
		k = len(self.world.params['b'])
		kr = k * r
		bs = np.array([float(f) for f in self.world.params['b']])
		b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
		kfunc = Automaton.kernel_core[(self.world.params.get('kn') or self.kn) - 1]
		return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b

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
		self.gpu_api = self.gpu_thr = self.gpu_fft = self.gpu_fftshift = None
		try:
			self.gpu_api = reikna.cluda.any_api()
			self.gpu_thr = self.gpu_api.Thread.create()
			self.gpu_fft = reikna.fft.FFT(A.astype(np.complex64)).compile(self.gpu_thr)
			self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
		except Exception as exc:
			# if str(exc) == "No supported GPGPU APIs found":
			self.has_gpu = False
			self.is_gpu = False
			print(exc)
			# raise exc

	def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
		if self.is_gpu and self.gpu_thr and gpu_func:
			op_dev = self.gpu_thr.to_device(A.astype(dtype))
			gpu_func(op_dev, op_dev, **kwargs)
			return op_dev.get()
		else:
			return cpu_func(A)
			# return np.roll(potential_shifted, (MIDX, MIDY), (1, 0))

	def fft(self, A): return self.run_gpu(A, np.fft.fft2, self.gpu_fft, np.complex64)
	def ifft(self, A): return self.run_gpu(A, np.fft.ifft2, self.gpu_fft, np.complex64, inverse=True)
	def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)

	def calc_once(self):
		A = self.world.cells
		self.world_FFT = self.fft(A)
		self.potential_FFT = self.kernel_FFT * self.world_FFT
		self.potential = self.fftshift(np.real(self.ifft(self.potential_FFT)))
		gfunc = Automaton.field_func[(self.world.params.get('gn') or self.gn) - 1]
		self.field = gfunc(self.potential, self.world.params['m'], self.world.params['s'])
		dt = 1 / self.world.params['T']
		if self.is_multi_step and self.field_old:
			D = 1/2 * (3 * self.field - self.field_old)
			self.field_old = self.field.copy()
		else:
			D = self.field
		if not self.is_soft_clip:
			A_new = np.clip(A + dt * D, 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
		else:
			A_new = Automaton.soft_clip(A + dt * D, 0, 1, 1/dt)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
		self.change = (A_new - A) / dt
		self.world.cells = A_new
		self.gen += 1
		self.time = round(self.time + dt, ROUND)
		if self.is_gpu:
			self.gpu_thr.synchronize()

	def calc_kernel(self):
		I, J = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
		self.X = (I - MIDX) / self.world.params['R']
		self.Y = (J - MIDY) / self.world.params['R']
		self.D = np.sqrt(self.X**2 + self.Y**2)

		self.kernel = self.kernel_shell(self.D)
		self.kernel_sum = np.sum(self.kernel)
		kernel_norm = self.kernel / self.kernel_sum
		self.kernel_FFT = self.fft(kernel_norm)
		self.kernel_updated = False

	def reset(self):
		self.gen = 0
		self.time = 0
		self.field_old = None

class Analyzer:
	STAT_NAMES = {'p_m':'Param m', 'p_s':'Param s', 'n':'Gen (#)', 't':'Time (s)', 
		'm':'Mass (mg)', 'g':'Growth (mg/s)', 'I':'Moment of inertia', 
		'd':'Mass-growth distance (mm)', 's':'Speed (mm/s)', 'w':'Angular speed (deg/s)', 'm_a':'Mass asymmetry (mg)'}
		# 'a':'Semi-major axis (mm)', 'b':'Semi-minor axis (mm)', 'e':'Eccentricity', 'c':'Compactness', 'w_th':'Shape angular speed (deg/s)'}
	STAT_HEADERS = ['p_m', 'p_s', 'n', 't', 'm', 'g', 'I', 'd', 's', 'w', 'm_a']
		# , 'a', 'b', 'e', 'c', 'w_th']
	SEGMENT_LEN = 200

	def __init__(self, automaton):
		self.automaton = automaton
		self.world = self.automaton.world
		# self.aaa = self.world.cells
		self.reset()

	def reset(self):
		self.is_empty = False
		self.is_full = False
		self.mass = 0
		self.growth = 0
		self.inertia = 0
		self.m_last_center = None
		self.m_center = None
		self.g_center = None
		self.mg_dist = 0
		self.m_shift = 0
		self.m_last_angle = None
		self.m_angle = 0
		self.m_rotate = 0
		self.mass_asym = 0
		# self.shape_major_axis = 0
		# self.shape_minor_axis = 0
		# self.shape_eccentricity = 0
		# self.shape_compactness = 0
		# self.shape_last_angle = None
		# self.shape_angle = 0
		# self.shape_rotate = 0
		self.series = []
		self.last_shift_idx = np.zeros(2)
		self.total_shift_idx = np.zeros(2)
		self.is_clip_segment = True

	def calc_stat(self):
		R, T = [self.world.params[k] for k in ('R', 'T')]
		A = self.world.cells
		G = np.maximum(self.automaton.field, 0)
		h, w = A.shape
		X, Y = self.automaton.X, self.automaton.Y
		m00 = self.mass = np.sum(A)
		g00 = self.growth = np.sum(G)
		self.is_empty = (self.mass < EPSILON)
		self.is_full = (np.sum(A[0,:]) + np.sum(A[h-1,:]) + np.sum(A[:,0]) + np.sum(A[:,w-1]) > 0)

		self.m_last_center = self.m_center
		self.m_last_angle = self.m_angle
		# self.shape_last_angle = self.shape_angle

		self.inertia = 0
		self.m_center = None
		self.g_center = None
		self.mg_dist = 0
		self.m_shift = 0
		self.m_angle = 0
		self.m_rotate = 0
		self.mass_asym = 0
		# self.shape_major_axis = 0
		# self.shape_minor_axis = 0
		# self.shape_eccentricity = 0
		# self.shape_compactness = 0
		# self.shape_angle = 0
		# self.shape_rotate = 0 
		if m00 > EPSILON:
			AX, AY = A*X, A*Y
			m01, m10 = np.sum(AX), np.sum(AY)
			m02, m20 = np.sum(AX*X), np.sum(AY*Y)
			mx, my = self.m_center = np.array([m01, m10]) / m00
			mu02, mu20 = m02 - my * m01, m20 - mx * m10
			self.inertia = (mu20 + mu02) / m00**2
			
			# m11 = np.sum(AY*X)
			# mu11 = m11 - mx * m01
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

			if g00 > EPSILON:
				g01, g10 = np.sum(G*X), np.sum(G*Y)
				gx, gy = self.g_center = np.array([g01, g10]) / g00
				self.mg_dist = np.linalg.norm(self.m_center - self.g_center)

			if self.m_last_center is not None and self.m_last_angle is not None:
				dm = self.m_center - self.m_last_center + self.last_shift_idx / R
				self.m_shift = np.linalg.norm(dm)
				self.m_angle = np.degrees(np.arctan2(dm[1], dm[0])) if self.m_shift >= EPSILON else 0
				self.m_rotate = self.m_angle - self.m_last_angle
				self.m_rotate = (self.m_rotate + 540) % 360 - 180
				if self.automaton.gen <= 2:
					self.m_rotate = 0

				midpoint = np.array([MIDX, MIDY])
				X, Y = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
				x0, y0 = self.m_last_center * R + midpoint - self.last_shift_idx
				x1, y1 = self.m_center * R + midpoint
				sign = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0)
				self.mass_asym = np.sum(A[sign>0]) - np.sum(A[sign<0])
				# self.aaa = A.copy(); self.aaa[sign<0] = 0

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
	def clear_series(self):
		self.series = []

	def add_stat(self):
		R, T, pm, ps = [self.world.params[k] for k in ('R', 'T', 'm', 's')]
		v = [pm, ps, self.automaton.gen, self.automaton.time, 
			self.mass/R/R, self.growth/R/R, self.inertia, 
			self.m_shift*T, self.m_rotate*T, self.mg_dist, self.mass_asym/R/R]
			# self.shape_major_axis, self.shape_minor_axis,
			# self.shape_eccentricity, self.shape_compactness, self.shape_rotate]
		if self.series == []:
			self.new_segment()
		segment = self.series[-1]
		segment.append(v)
		if self.is_clip_segment:
			while len(segment) > self.SEGMENT_LEN:
				segment.pop(0)

	def center_world(self):
		if self.mass < EPSILON or self.m_center is None:
			return
		self.last_shift_idx = (self.m_center * self.world.params['R']).astype(int)
		self.world.cells = np.roll(self.world.cells, -self.last_shift_idx, (1, 0))
		self.total_shift_idx += self.last_shift_idx

	def recurrence_plot(self, e=0.1, steps=10):
		''' https://stackoverflow.com/questions/33650371/recurrence-plot-in-python '''
		d = scipy.spatial.distance.pdist(self.series[:, None])
		d = np.floor(d/e)
		d[d>steps] = steps
		Z = scipy.spatial.distance.squareform(d)
		return Z

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
			except FileNotFoundError:
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
			except FileNotFoundError:
				self.video = None
				STATUS.append("> no ffmpeg program found!")
		else:
			if self.video:
				self.video.stdin.close()
				STATUS.append("> video saved to '" + self.video_path + "'")
		self.gif[0].save(self.gif_path, format=self.GIF_EXT.lstrip('.'), save_all=True, append_images=self.gif[1:], loop=0, duration=1000//self.ANIM_FPS)
		self.gif = None
		STATUS.append("> GIF saved to '" + self.gif_path + "'")
		self.is_recording = False

class Lenia:
	MARKER_COLORS = [95,95,95,127,127,127,255,255,255]
	def __init__(self):
		self.is_run = True
		self.is_once = False
		self.is_show = True
		self.is_closing = False
		self.show_what = 0
		self.is_show_markers = True
		self.stats_mode = 0
		self.stat_x = 4
		self.stat_y = 5
		self.is_show_fps = False
		self.fps = None
		self.last_time = None
		self.fore = None
		self.back = None
		self.is_layered = False
		self.is_auto_center = False
		self.is_auto_load = False
		self.trace_dir = 0
		self.trace_small = False
		''' http://hslpicker.com/ '''
		self.colormaps = [
			self.create_colormap(np.array([[0,0,4],[0,0,8],[0,4,8],[0,8,8],[4,8,4],[8,8,0],[8,4,0],[8,0,0],[4,0,0]])), #BCYR
			self.create_colormap(np.array([[0,2,0],[0,4,0],[4,6,0],[8,8,0],[8,4,4],[8,0,8],[4,0,8],[0,0,8],[0,0,4]])), #GYPB
			self.create_colormap(np.array([[4,0,2],[8,0,4],[8,0,6],[8,0,8],[4,4,4],[0,8,0],[0,6,0],[0,4,0],[0,2,0]])), #PPGG
			self.create_colormap(np.array([[4,4,6],[2,2,4],[2,4,2],[4,6,4],[6,6,4],[4,2,2]])), #BGYR
			self.create_colormap(np.array([[4,6,4],[2,4,2],[4,4,2],[6,6,4],[6,4,6],[2,2,4]])), #GYPB
			self.create_colormap(np.array([[6,6,4],[4,4,2],[4,2,4],[6,4,6],[4,6,6],[2,4,2]])), #YPCG
			self.create_colormap(np.array([[0,0,0],[3,3,3],[4,4,4],[5,5,5],[8,8,8]]))] #B/W
		self.colormap_id = 0
		self.set_colormap()
		self.last_key = None
		self.excess_key = None
		self.update = None
		self.clear_job = None
		self.is_save_image = False

		self.read_animals()
		self.world = Board((SIZEY, SIZEX))
		self.automaton = Automaton(self.world)
		self.analyzer = Analyzer(self.automaton)
		self.recorder = Recorder(self.world)
		self.clear_transform()
		self.create_window()
		self.create_menu()

	def clear_transform(self):
		self.tx = {'shift':[0, 0], 'rotate':0, 'R':self.world.params['R'], 'flip':-1}

	def read_animals(self):
		with open('animals.json', encoding='utf-8') as file:
			self.animal_data = json.load(file)

	def load_animal_id(self, id, **kwargs):
		self.animal_id = max(0, min(len(self.animal_data)-1, id))
		self.load_part(Board.from_data(self.animal_data[self.animal_id]), **kwargs)

	def load_animal_code(self, code, **kwargs):
		if not code: return
		id = self.get_animal_id(code)
		if id: self.load_animal_id(id, **kwargs)

	def get_animal_id(self, code):
		code_sp = code.split(':')
		n = int(code_sp[1]) if len(code_sp)==2 else 1
		it = (id for (id, data) in enumerate(self.animal_data) if data["code"]==code_sp[0])
		for i in range(n):
			id = next(it, None)
		return id

	def load_part(self, part, is_replace=True, is_random=False, is_auto_load=False, repeat=1):
		self.fore = part
		if part.names[0].startswith('~'):
			part.names[0] = part.names[0].lstrip('~')
			self.world.params['R'] = part.params['R']
			self.automaton.calc_kernel()
		if is_replace:
			self.world.names = part.names.copy()
		if part.params is not None and part.cells is not None:
			is_life = ((self.world.params.get('kn') or self.automaton.kn) == 4)
			will_be_life = ((part.params.get('kn') or self.automaton.kn) == 4)
			if not is_life and will_be_life:
				self.colormap_id = len(self.colormaps) - 1
				self.win.title('Conway\'s Game of Life')
			elif is_life and not will_be_life:
				self.colormap_id = 0
				self.world.params['R'] = DEF_R
				self.automaton.calc_kernel()
				self.win.title('Lenia')
			if self.is_layered:
				self.back = copy.deepcopy(self.world)
			if is_replace and not self.is_layered:
				if not is_auto_load:
					self.world.params = {**part.params, 'R':self.world.params['R']}
					self.automaton.calc_kernel()
				self.world.clear()
				self.automaton.reset()
				if not is_auto_load:
					self.analyzer.reset()
			self.clear_transform()
			for i in range(repeat):
				if is_random:
					self.tx['rotate'] = np.random.random() * 360
					h1, w1 = self.world.cells.shape
					h, w = min(part.cells.shape, self.world.cells.shape)
					self.tx['shift'] = [np.random.randint(d1 + d) - d1//2 for (d,d1) in [(h,h1), (w,w1)]]
					self.tx['flip'] = np.random.randint(3) - 1
				self.world.add_transformed(part, self.tx)

	def check_auto_load(self):
		if self.is_auto_load:
			self.load_part(self.fore, is_auto_load=True)

	def transform_world(self):
		if self.is_layered:
			self.world.cells = self.back.cells.copy()
			self.world.params = self.back.params.copy()
			self.world.transform(self.tx, mode='Z', is_world=True)
			self.world.add_transformed(self.fore, self.tx)
		else:
			if not self.is_run:
				if self.back is None:
					self.back = copy.deepcopy(self.world)
				else:
					self.world.cells = self.back.cells.copy()
					self.world.params = self.back.params.copy()
			self.world.transform(self.tx, is_world=True)
		self.automaton.calc_kernel()

	def clear_world(self):
		self.world.clear()
		if self.is_layered:
			self.back = copy.deepcopy(self.world)
		self.automaton.reset()
		self.analyzer.reset()

	def random_world(self):
		self.world.clear()
		border = self.world.params['R']
		rand = np.random.rand(SIZEY - border*2, SIZEX - border*2)
		self.world.add(Board.from_values(None, None, rand))
		if self.is_layered:
			self.back = copy.deepcopy(self.world)
		self.automaton.reset()
		self.analyzer.reset()

	def toggle_trace(self, dir, small):
		if self.trace_dir == 0:
			self.trace_dir = dir
			self.trace_small = small
			self.is_auto_center = True
			self.is_auto_load = True
		else:
			self.trace_dir = 0

	def stop_trace(self):
		self.trace_dir = 0

	def trace_params(self):
		s = 's+' if self.trace_small else ''
		if self.trace_dir == +1:
			if self.analyzer.is_empty: self.key_press_internal(s+'w')
			elif self.analyzer.is_full: self.key_press_internal(s+'q')
		elif self.trace_dir == -1:
			if self.analyzer.is_empty: self.key_press_internal(s+'a')
			elif self.analyzer.is_full: self.key_press_internal(s+'s')

	def create_window(self):
		self.win = tk.Tk()
		self.win.title('Lenia')
		self.win.bind('<Key>', self.key_press_event)
		self.frame = tk.Frame(self.win, width=SIZEX*PIXEL, height=SIZEY*PIXEL)
		self.frame.pack()
		self.canvas = tk.Canvas(self.frame, width=SIZEX*PIXEL, height=SIZEY*PIXEL)
		self.canvas.place(x=-1, y=-1)
		self.panel1 = self.create_panel(0, 0)
		# self.panel2 = self.create_panel(1, 0)
		# self.panel3 = self.create_panel(0, 1)
		# self.panel4 = self.create_panel(1, 1)
		self.info = tk.Label(self.win)
		self.info.pack()

	def create_panel(self, c, r):
		buffer = np.uint8(np.zeros((SIZEY*PIXEL,SIZEX*PIXEL)))
		img = Image.frombuffer('P', (SIZEX*PIXEL,SIZEY*PIXEL), buffer, 'raw', 'P', 0, 1)
		photo = ImageTk.PhotoImage(image=img)
		return self.canvas.create_image(c*SIZEY, r*SIZEX, image=photo, anchor=tk.NW)

	def create_colormap(self, colors):
		nval = 256 - 3
		ncol = colors.shape[0]
		colors = np.vstack((colors, np.array([[0,0,0]])))
		v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
		i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
		k = v / (nval-1) * (ncol-1)  # interpolate between 0 .. ncol-1
		k1 = k.astype(int)
		c1, c2 = colors[k1,i], colors[k1+1,i]
		c = (k-k1) * (c2-c1) + c1  # interpolate between c1 .. c2
		return np.rint(c / 8 * 255).astype(int).tolist() + self.MARKER_COLORS

	def set_colormap(self):
		self.colormap_demo = np.tile(np.arange(SIZEX), (1, SIZEY)) / SIZEX

	SHOW_WHAT_NUM = 7
	def update_win(self):
		if self.stats_mode in [0, 3]:
			change_range = 1 if not self.automaton.is_soft_clip else 1.4
			if self.show_what==0: self.draw_world(self.world.cells, 0, 1, is_shift=True, is_shift_zero=True, markers=['arrow','ruler','grid'])
			# if self.show_what==0: self.draw_world(self.analyzer.aaa, 0, 1, is_shift=True, is_shift_zero=True, markers=['arrow','ruler','grid'])
			elif self.show_what==1: self.draw_world(self.automaton.potential, 0, 2*self.world.params['m'], is_shift=True, is_shift_zero=True, markers=['arrow','ruler','grid'])
			elif self.show_what==2: self.draw_world(self.automaton.field, -1, 1, is_shift=True, markers=['arrow','ruler','grid'])
			elif self.show_what==3: self.draw_world(self.automaton.change, -change_range, change_range, is_shift=True, markers=['arrow','ruler','grid'])
			elif self.show_what==4: self.draw_world(self.automaton.kernel, 0, 1, markers=['ruler','fixgrid'])
			elif self.show_what==5: self.draw_world(self.automaton.fftshift(np.log(np.absolute(self.automaton.world_FFT))), 0, 5)
			elif self.show_what==6: self.draw_world(self.automaton.fftshift(np.log(np.absolute(self.automaton.potential_FFT))), 0, 5)
			elif self.show_what==7: self.draw_world(self.colormap_demo, 0, 1)
			self.img.putpalette(self.colormaps[self.colormap_id])
		elif self.stats_mode in [1, 2]:
			self.draw_black()

		if self.stats_mode in [1, 2, 3]:
			self.draw_stats()

		if self.recorder.is_recording and self.is_run:
			self.recorder.record_frame(self.img)
		if self.is_save_image:
			self.recorder.save_image(self.img, filename='saved')
			self.is_save_image = False

		photo = ImageTk.PhotoImage(image=self.img)
		# photo = tk.PhotoImage(width=SIZEX, height=SIZEY)
		self.canvas.itemconfig(self.panel1, image=photo)
		self.win.update()

	def draw_world(self, A, vmin=0, vmax=1, is_shift=False, is_shift_zero=False, markers=[]):
		if is_shift and not self.is_auto_center:
			A = np.roll(A, self.analyzer.total_shift_idx.astype(int), (1, 0))
		if is_shift_zero and self.automaton.is_soft_clip:
			if vmin==0: vmin = np.amin(A)
		buffer = np.uint8(np.clip((A-vmin) / (vmax-vmin), 0, 1) * 252)  # .copy(order='C')
		if self.is_show_markers and ('grid' in markers or 'fixgrid' in markers):
			self.draw_grid(buffer, is_fixed='fixgrid' in markers)
		buffer = np.repeat(np.repeat(buffer, PIXEL, axis=0), PIXEL, axis=1)
		zero = np.uint8(np.clip((0-vmin) / (vmax-vmin), 0, 1) * 252)
		for i in range(PIXEL_BORDER):
			buffer[i::PIXEL, :] = zero;  buffer[:, i::PIXEL] = zero
		self.img = Image.frombuffer('P', (SIZEX*PIXEL,SIZEY*PIXEL), buffer, 'raw', 'P', 0, 1)
		if self.is_show_markers and ('arrow' in markers or 'ruler' in markers):
			self.draw_markers(markers)

	def draw_black(self):
		size = (SIZEX*PIXEL,SIZEY*PIXEL)
		self.img = Image.frombuffer('L', size, np.zeros(size), 'raw', 'L', 0, 1)

	def draw_grid(self, buffer, is_fixed=False):
		R = self.world.params['R']
		n = R // 40 if R >= 15 else -1
		for i in range(-n, n+1):
			sx, sy = 0, 0
			if self.is_auto_center and not is_fixed:
				sx, sy = (self.analyzer.total_shift_idx).astype(int)
			grid = buffer[(MIDY - sy + i) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R];  grid[grid==0] = 253
			grid = buffer[(MIDY - sy) % R:SIZEY:R, (MIDX - sx + i) % R:SIZEX:R];  grid[grid==0] = 253

	def draw_markers(self, markers=[]):
		R, T = [self.world.params[k] for k in ('R', 'T')]
		midpoint = np.array([MIDX, MIDY])
		draw = ImageDraw.Draw(self.img)
		d2 = np.array([1, 1]) * 2
		if 'arrow' in markers and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
			shift = self.analyzer.total_shift_idx if not self.is_auto_center else np.zeros(2)
			m0 = self.analyzer.m_last_center * R + midpoint + shift - self.analyzer.last_shift_idx
			m1 = self.analyzer.m_center * R + midpoint + shift
			ms = m1 % np.array([SIZEX, SIZEY]) - m1
			m2, m3 = [m0 + (m1 - m0) * n * T for n in [1,2]]
			for i in range(-1, 2):
				for j in range(-1, 2):
					adj = np.array([i*SIZEX, j*SIZEY]) + ms
					draw.line(tuple((m0+adj)*PIXEL) + tuple((m3+adj)*PIXEL), fill=254, width=1)
					[draw.ellipse(tuple((m+adj-d2)*PIXEL) + tuple((m+adj+d2)*PIXEL), fill=c) for (m,c) in [(m0,254),(m1,255),(m2,255),(m3,255)]]
		if 'ruler' in markers:
			x0, y0 = SIZEX*PIXEL-20, SIZEY*PIXEL-20
			x1, x2, x3, y1, y2 = x0-10-R*PIXEL, x0-10-R*PIXEL//2, x0-10, y0+3, y0+8
			draw.text((x0, y0), 'R', fill=254)
			draw.line([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x2,y2),(x3,y2),(x3,y1)], fill=254, width=1)
		del draw

	def draw_stats(self):
		draw = ImageDraw.Draw(self.img)
		series = self.analyzer.series
		if series != [] and self.stats_mode in [2, 3]:
			series = [series[-1]]
		if series != [] and series != [[]]:
			X = [[v[self.stat_x] for v in s] for s in series]
			Y = [[v[self.stat_y] for v in s] for s in series]
			# if self.stat_x in [0,1] or self.stat_y in [0,1]:
			# 	X, Y = X[-100:], Y[-100:]
			if self.stat_x in [2, 3]: X = [[x - min(s) for x in s] for s in X]
			if self.stat_y in [2, 3]: Y = [[y - min(s) for y in s] for s in Y]
			xmin, xmax = min(min(s) for s in X if s), max(max(s) for s in X if s)
			ymin, ymax = min(min(s) for s in Y if s), max(max(s) for s in Y if s)
			if xmax-xmin>EPSILON and ymax-ymin>EPSILON:
				if self.stats_mode in [1, 2]:
					C = reversed([194 // 2**i + 61 for i in range(len(X))])
				else:
					C = [255] * len(X)
				for x, y, c in zip(X, Y, C):
					xa, ya = np.array(x), np.array(y)
					xa = (xa - xmin) / (xmax - xmin) * (SIZEX - 10) + 5
					ya = (ya - ymin) / (ymax - ymin) * (SIZEY - 10) + 5
					draw.line(list(zip(xa, ya)), fill=c, width=1)
		del draw

	def calc_fps(self):
		if self.automaton.gen == 0:
			self.last_time = time.time()
		elif self.automaton.gen % FPS_FREQ == 0:
			this_time = time.time()
			self.fps = FPS_FREQ / (this_time - self.last_time)
			self.last_time = this_time

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

	ANIMAL_KEY_LIST = {'1':'O2(a)', '2':'OG2', '3':'OV2', '4':'P4(a)', '5':'2S1:5', '6':'2S2:2', '7':'P6,3s', '8':'2PG1:2', '9':'3H3', '0':'~gldr', \
		's+1':'3G:4', 's+2':'3GG', 's+3':'K5(4,1)', 's+4':'K7(4,3)', 's+5':'K9(5,4)', 's+6':'3A5', 's+7':'4A6', 's+8':'2D10', 's+9':'4F12', 's+0':'~ggun', \
		'c+1':'4Q(5,5,5,5):3', 'c+2':'2P7:2', 'c+3':'3GA', 'c+4':'K4(2,2):3', 'c+5':'K4(2,2):5', 'c+6':'3R4(3,3,2):4', 'c+7':'3F6', 'c+8':'4F7', 'c+9':'', 'c+0':'bbug'}
	def process_key(self, k):
		global STATUS
		inc_or_dec     = 1 if 's+' not in k else -1
		inc_10_or_1  = (10 if 's+' not in k else 1) if 'c+' not in k else 0 
		inc_big_or_not = 0 if 'c+' not in k else 1
		inc_1_or_10    = 1 if 's+' not in k else 10
		inc_mul_or_not = 1 if 's+' not in k else 0
		double_or_not  = 2 if 's+' not in k else 1
		inc_or_not     = 0 if 's+' not in k else 1

		is_ignore = False
		if not self.is_internal_key:
			self.stop_trace()

		if k in ['escape']: self.is_closing = True; self.close()
		elif k in ['enter', 'return']: self.is_run = not self.is_run
		elif k in [' ', 'space']: self.is_once = not self.is_once; self.is_run = False
		elif k in ['quoteright']: self.is_show = not self.is_show
		elif k in ['quoteleft', 's+quoteleft']: self.colormap_id = (self.colormap_id + inc_or_dec) % len(self.colormaps); self.set_colormap()
		elif k in ['tab', 's+tab']: self.show_what = (self.show_what + inc_or_dec) % self.SHOW_WHAT_NUM
		elif k in ['c+tab']: self.show_what = 0 if self.show_what == self.SHOW_WHAT_NUM else self.SHOW_WHAT_NUM
		elif k in ['q', 's+q']: self.world.params['m'] += inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.update = 'param'
		elif k in ['a', 's+a']: self.world.params['m'] -= inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.update = 'param'
		elif k in ['w', 's+w']: self.world.params['s'] += inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.update = 'param'
		elif k in ['s', 's+s']: self.world.params['s'] -= inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.update = 'param'
		elif k in ['t', 's+t']: self.world.params['T'] = max(1, self.world.params['T'] *  double_or_not + inc_or_not); self.update = 'param'
		elif k in ['g', 's+g']: self.world.params['T'] = max(1, self.world.params['T'] // double_or_not - inc_or_not); self.update = 'param'
		elif k in ['r', 's+r']: self.tx['R'] = max(1, self.tx['R'] + inc_10_or_1); self.transform_world(); self.update = 'param'
		elif k in ['f', 's+f']: self.tx['R'] = max(1, self.tx['R'] - inc_10_or_1); self.transform_world(); self.update = 'param'
		elif k in ['c+q', 's+c+q']: self.toggle_trace(+1, 's+' in k)
		elif k in ['c+a', 's+c+a']: self.toggle_trace(-1, 's+' in k)
		elif k in ['c+w', 's+c+w']: pass  # randam params and/or peaks
		elif k in ['c+r']: self.tx['R'] = DEF_R; self.transform_world(); self.update = 'param'
		elif k in ['c+f']: self.tx['R'] = self.fore.params['R'] if self.fore else DEF_R; self.transform_world(); self.update = 'param'
		elif k in ['c+y', 's+c+y']: self.automaton.kn = (self.automaton.kn + inc_or_dec - 1) % len(self.automaton.kernel_core) + 1; self.update = 'kn'
		elif k in ['c+u', 's+c+u']: self.automaton.gn = (self.automaton.gn + inc_or_dec - 1) % len(self.automaton.field_func) + 1; self.update = 'gn'
		elif k in ['c+i']: self.automaton.is_soft_clip = not self.automaton.is_soft_clip
		elif k in ['c+o']: self.automaton.is_multi_step = not self.automaton.is_multi_step
		elif k in ['c+p']: self.automaton.is_inverted = not self.automaton.is_inverted; self.world.params['T'] *= -1; self.world.params['m'] = 1 - self.world.params['m']; self.world.cells = 1 - self.world.cells
		elif k in ['down',  's+down',  'c+down' ]: self.tx['shift'][0] += inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
		elif k in ['up',    's+up',    'c+up'   ]: self.tx['shift'][0] -= inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
		elif k in ['right', 's+right', 'c+right']: self.tx['shift'][1] += inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
		elif k in ['left',  's+left',  'c+left' ]: self.tx['shift'][1] -= inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
		elif k in ['pageup',   's+pageup',   'c+pageup',   'prior', 's+prior', 'c+prior']: self.tx['rotate'] += inc_10_or_1 + inc_big_or_not * 45; self.transform_world()
		elif k in ['pagedown', 's+pagedown', 'c+pagedown', 'next',  's+next' , 'c+next' ]: self.tx['rotate'] -= inc_10_or_1 + inc_big_or_not * 45; self.transform_world()
		elif k in ['home'   ]: self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
		elif k in ['end'    ]: self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
		elif k in ['equal'  ]: self.tx['flip'] = 2 if self.tx['flip'] != 0 else -1; self.transform_world()
		elif k in ['s+equal']: self.tx['flip'] = 3 if self.tx['flip'] != 0 else -1; self.transform_world()
		elif k in ['m']: self.is_auto_center = not self.is_auto_center
		elif k in ['backspace', 'delete']: self.clear_world()
		elif k in ['c', 's+c']: self.load_animal_id(self.animal_id - inc_1_or_10); self.update = 'animal'
		elif k in ['v', 's+v']: self.load_animal_id(self.animal_id + inc_1_or_10); self.update = 'animal'
		elif k in ['z']: self.load_animal_id(self.animal_id); self.update = 'animal'
		elif k in ['x', 's+x']: self.load_part(self.fore, is_random=True, is_replace=False, repeat=inc_1_or_10)
		elif k in ['b']: self.random_world()
		elif k in ['n']: pass # random last seed
		elif k in ['c+z']: self.is_auto_load = not self.is_auto_load
		elif k in ['c+x']: self.is_layered = not self.is_layered
		elif k in ['c+c', 's+c+c', 'c+s', 's+c+s']:
			A = copy.deepcopy(self.world)
			A.crop()
			data = A.to_data(is_shorten='s+' not in k)
			if k.endswith('c'):
				self.clipboard_st = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
				self.win.clipboard_clear()
				self.win.clipboard_append(self.clipboard_st)
				# print(self.clipboard_st)
				STATUS.append("> board saved to clipboard")
			elif k.endswith('s'):
				with open('saved.rle', 'w', encoding='utf8') as file:
					file.write('#N '+A.long_name()+'\n')
					file.write('x = '+str(A.cells.shape[1])+', y = '+str(A.cells.shape[0])+', rule = Lenia('+A.params2st()+')\n')
					file.write(data['cells'].replace('$','$\n')+'\n')
				data['cells'] = data['cells'].split('$')
				with open('saved.json', 'w', encoding='utf-8') as file:
					json.dump(data, file, indent=4, ensure_ascii=False)
				with open('saved.csv', 'w', newline='\n') as file:
					writer = csv.writer(file)
					writer.writerow([self.analyzer.stat_name(x=x) for x in self.analyzer.STAT_HEADERS])
					writer.writerows([e for l in self.analyzer.series for e in l])
				STATUS.append("> data and image saved to 'saved.*'")
				self.is_save_image = True
		elif k in ['c+v']:
			self.clipboard_st = self.win.clipboard_get()
			data = json.loads(self.clipboard_st)
			self.load_part(Board.from_data(data))
		elif k in ['c+d', 's+c+d']: self.recorder.toggle_recording(is_save_frames='s+' in k)
		elif k in ['c+g']:
			if self.automaton.has_gpu:
				self.automaton.is_gpu = not self.automaton.is_gpu
		elif k in [m+str(i) for i in range(10) for m in ['','s+','c+','s+c+']]: self.load_animal_code(self.ANIMAL_KEY_LIST.get(k)); self.update = 'animal'
		elif k in ['comma']: self.update = 'animal'
		elif k in ['period']: self.update = 'param'
		elif k in ['h']: self.is_show_markers = not self.is_show_markers
		elif k in ['c+h']: self.is_show_fps = not self.is_show_fps
		elif k in ['j', 's+j']: self.stats_mode = (self.stats_mode - inc_or_dec) % 4
		elif k in ['c+j']: self.analyzer.clear_segment()
		elif k in ['s+c+j']: self.analyzer.clear_series()
		elif k in ['k', 's+k']:
			if self.stats_mode == 0: self.stats_mode = 3
			while True:
				self.stat_x = (self.stat_x + inc_or_dec) % len(self.analyzer.STAT_HEADERS); self.update = 'stats'
				if self.stat_x != self.stat_y and self.stat_x > 2: break
		elif k in ['l', 's+l']:
			if self.stats_mode == 0: self.stats_mode = 3
			while True:
				self.stat_y = (self.stat_y + inc_or_dec) % len(self.analyzer.STAT_HEADERS); self.update = 'stats'
				if self.stat_x != self.stat_y and self.stat_y > 2: break
		elif k in ['c+k']: self.analyzer.is_clip_segment = not self.analyzer.is_clip_segment
		# elif k in ['slash']: m = self.menu.children[self.menu_values['animal'][0]].children['!menu']; m.post(self.win.winfo_rootx(), self.win.winfo_rooty())
		elif k.endswith('_l') or k.endswith('_r'): is_ignore = True
		else: self.excess_key = k

		if not is_ignore and self.is_loop:
			self.world.params = {k:round(x, ROUND) if type(x)==float else x for (k,x) in self.world.params.items()}
			self.tx = {k:round(x, ROUND) if type(x)==float else x for (k,x) in self.tx.items()}
			A = self.world.cells.copy()
			self.automaton.calc_once()
			# self.analyzer.calc_stat()
			self.world.cells = A
			self.update_menu()

	def get_acc_func(self, key, acc, animal_id=None):
		acc = acc if acc else key if key else None
		if acc: acc = acc.replace('s+','Shift+').replace('c+','Ctrl+').replace('m+','Cmd+').replace('a+','Slt+')
		if animal_id:
			func = lambda:self.load_animal_id(int(animal_id))
		else:
			func = lambda:self.key_press_internal(key.lower()) if key else None
		state = 'normal' if key or animal_id else tk.DISABLED
		return {'accelerator':acc, 'command':func, 'state':state}
	def create_submenu(self, parent, items):
		m = tk.Menu(parent, tearoff=True)
		m.seq = 0
		for i in items:
			m.seq += 1
			if i is None or i=='':
				m.add_separator()
			elif type(i) in [tuple, list]:
				m.add_cascade(label=i[0], menu=self.create_submenu(m, i[1]))
			else:
				first, text, key, acc, *_ = i.split('|') + ['']*2
				kind, name = first[:1], first[1:]
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
					m.add_command(label='', state=tk.DISABLED) # background='navy', foreground='white')
				elif kind=='&':
					m.add_command(label=text, **self.get_acc_func(key, acc, animal_id=name))
		return m
	def get_animal_nested_list(self):
		root = []
		stack = [root]
		id = 0
		for data in self.animal_data:
			code = data['code']
			if code.startswith('>'):
				next_level = int(code[1:])
				d = len(stack) - next_level
				for i in range(d):
					stack.pop()
				for i in range(max(-d, 0) + 1):
					new_list = ('{name} {cname}'.format(**data), [])
					stack[-1].append(new_list)
					stack.append(new_list[1])
			else:
				stack[-1].append('&{id}|{name} {cname}|'.format(id=id, **data))
			id += 1
		return root

	def get_nested_attr(self, name):
		obj = self
		for n in name.split('.'):
			obj = getattr(obj, n)
		return obj
	def update_menu_value(self, name, value):
		text = ''
		info = self.menu_values[name]
		self.menu.children[info[0]].entryconfig(info[1], label='{text} [{value}]'.format(text=text or info[2], value=value))
	def get_value_text(self, name):
		if name=='anm': return '#'+str(self.animal_id+1)+' '+self.world.long_name()
		elif name=='kn': return ["Polynomial","Exponential","Step","Staircase"][(self.world.params.get('kn') or self.automaton.kn) - 1]
		elif name=='gn': return ["Polynomial","Exponential","Step"][(self.world.params.get('gn') or self.automaton.gn) - 1]
		elif name=='clr': return ["Vivid blue/red","Vivid green/purple","Vivid red/green","Pale blue/red","Pale green/purple","Pale yellow/green","Black/white"][self.colormap_id]
		elif name=='shw': return ["World","Potential","Field","Change","Kernel","World FFT","Potential FFT","Colormap"][self.show_what]
		elif name=='stm': return ["None","All","Segment","Overlay"][self.stats_mode]
		elif name=='stx': return self.analyzer.stat_name(i=self.stat_x)
		elif name=='sty': return self.analyzer.stat_name(i=self.stat_y)
	def update_menu(self):
		for name in self.menu_vars:
			self.menu_vars[name].set(self.get_nested_attr(name))
		for (name, info) in self.menu_params.items():
			value = '['+Board.fracs2st(self.world.params[name])+']' if name=='b' else self.world.params[name]
			self.menu.children[info[0]].entryconfig(info[1], label='{text} ({param} = {value})'.format(text=info[2], param=name, value=value))
		for (name, info) in self.menu_values.items():
			value = self.get_value_text(name)
			self.menu.children[info[0]].entryconfig(info[1], label='{text} [{value}]'.format(text=info[2], value=value))

	PARAM_TEXT = {'m':'Field center', 's':'Field width', 'R':'Space units', 'T':'Time units', 'dr':'Space step', 'dt':'Time step', 'b':'Kernel peaks'}
	VALUE_TEXT = {'anm':'Animal', 'kn':'Kernel core', 'gn':'Field', 'shw':'Show', 'clr':'Colors'}
	def create_menu(self):
		self.menu_vars = {}
		self.menu_params = {}
		self.menu_values = {}
		self.menu = tk.Menu(self.win, tearoff=True)
		self.win.config(menu=self.menu)

		items2 = ['^automaton.is_gpu|Use GPU|c+G' if self.automaton.has_gpu else '|No GPU available|']
		self.menu.add_cascade(label='Main', menu=self.create_submenu(self.menu, [
			'^is_run|Start|Return', '|Once|Space'] + items2 + [None,
			'^is_show|Show|Quoteright|\'', '@shw|Show|Tab', '|Show colormap|c+Tab', '@clr|Colors|QuoteLeft|`', None,
			'^is_show_markers|Show markers|H', '^is_show_fps|Show FPS|c+H', None,
			'@stm|Show stats|J', '@stx|Stats X axis|K', '@sty|Stats Y axis|L', 
			'|Clear segment|c+J', '|Clear all|s+c+J', '^analyzer.is_clip_segment|Clip segments|c+K', None, 
			'|Save data & image|c+S', '|Save with expanded format|s+c+S', 
			'^recorder.is_recording|Record video & gif|c+D', '|Record with frames saved|s+c+D', None,
			'|Quit|Escape']))

		self.menu.add_cascade(label='View', menu=self.create_submenu(self.menu, [
			'^is_auto_center|Auto center|c+M', None,
			'|(Small adjust)||s+Up', '|(Large adjust)||m+Up',
			'|Move up|Up', '|Move down|Down', '|Move left|Left', '|Move right|Right',
			'|Rotate clockwise|PageUp', '|Rotate anti-clockwise|PageDown', None,
			'|Flip vertically|Home', '|Flip horizontally|End',
			'|Mirror horizontally|Equal|=', '|Mirror flip|s+Equal|+']))

		items2 = []
		# for (key, code) in self.ANIMAL_KEY_LIST.items():
			# id = self.get_animal_id(code)
			# if id: items2.append('|{name} {cname}|{key}'.format(**self.animal_data[id], key=key))
		self.menu.add_cascade(label='Animal', menu=self.create_submenu(self.menu, [
			'|Show animal name|Comma|,', '|Place at center|Z', '|Place at random|X',
			'|Previous animal|C', '|Next animal|V', '|Previous 10|s+C', '|Next 10|s+V', None,
			'|Shortcuts 1-10|1', '|Shortcuts 11-20|s+1', '|Shortcuts 21-30|c+1', None,
			('Full list', self.get_animal_nested_list())]))

		self.menu.add_cascade(label='World', menu=self.create_submenu(self.menu, [
			'|Clear|Backspace', '|Random|B', '|Random (last seed)|N', None,
			'^is_auto_load|Auto put (place/paste/random)|c+Z', '^is_layered|Layer mode|c+X', None,
			'|Copy|c+C', '|Paste|c+V']))

		items2 = ['|Fewer peaks|BracketLeft|[', '|More peaks|BracketRight|]', None]
		for i in range(5):
			items2.append('|Taller peak {n}|{key}'.format(n=i+1, key='YUIOP'[i]))
			items2.append('|Shorter peak {n}|{key}'.format(n=i+1, key='s+'+'YUIOP'[i]))
		items2 += [None, '|Random peaks & field|s+c+W']

		# '@anm||', '#m|Field center', '#s|Field width', '#R|Space units', '#T|Time units', '#b|Kernel peaks', 
		self.menu.add_cascade(label='Params', menu=self.create_submenu(self.menu, [
			'|Show params|Period|.', None,
			'|(Small adjust)||s+Q', '|Higher field (m + 0.01)|Q', '|Lower field (m - 0.01)|A',
			'|Wider field (s + 0.001)|W', '|Narrower field (s - 0.001)|S', None,
			'|Bigger size (R + 10)|R', '|Smaller size (R - 10)|F',
			'|Slower speed (T * 2)|T', '|Faster speed (T / 2)|G', None,
			('Peaks', items2), None,
			'|Trace field higher|c+Q', '|Trace field lower|c+A', 
			'|Random field|c+W', '|Reset size|c+R', '|Animal\'s original size|c+F', None,
			'@kn|Kernel|c+Y', '@gn|Field|c+U',
			'^automaton.is_soft_clip|Use soft clip|c+I', '^automaton.is_multi_step|Use multi-step|c+O', 
			'^automaton.is_inverted|Invert|c+P']))

	def update_info(self):
		global STATUS
		if self.excess_key:
			print(self.excess_key)
			self.excess_key = None
		if self.update or STATUS or self.is_show_fps:
			info_st = ""
			if STATUS: info_st = "\n".join(STATUS)
			elif self.is_show_fps and self.fps: info_st = 'FPS: {0:.1f}'.format(self.fps)
			elif self.update == 'param': info_st = self.world.params2st()
			elif self.update == 'animal': info_st = self.world.long_name()
			elif self.update == 'stats': info_st = 'X axis: {0}, Y axis: {1}'.format(self.analyzer.stat_name(i=self.stat_x), self.analyzer.stat_name(i=self.stat_y))
			elif self.update in self.menu_values: info_st = "{text} [{value}]".format(text=self.VALUE_TEXT[self.update], value=self.get_value_text(self.update))
			self.info.config(text=info_st)
			STATUS = []
			self.update = None
			if self.clear_job is not None:
				self.win.after_cancel(self.clear_job)
			self.clear_job = self.win.after(5000, self.clear_info)

	def clear_info(self):
		self.info.config(text="")
		self.clear_job = None

	def loop(self):
		self.is_loop = True
		self.win.after(0, self.run)
		self.win.protocol('WM_DELETE_WINDOW', self.close)
		self.win.mainloop()

	def close(self):
		self.is_loop = False
		if self.recorder.is_recording:
			self.recorder.finish_record()
		self.win.destroy()

	def run(self):
		while self.is_loop:
			if self.last_key:
				self.process_key(self.last_key)
				self.last_key = None
			if self.is_closing:
				break
			if self.is_run or self.is_once:
				self.calc_fps()
				self.automaton.calc_once()
				self.analyzer.center_world()
				self.analyzer.calc_stat()
				self.analyzer.add_stat()
				if not self.is_layered:
					self.back = None
					self.clear_transform()
				if self.trace_dir != 0:
					self.trace_params()
				self.is_once = False
			self.update_info()
			if self.is_show:
				self.update_win()
			else:
				self.win.update()

if __name__ == '__main__':
	lenia = Lenia()
	lenia.load_animal_code(lenia.ANIMAL_KEY_LIST['2'])
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
