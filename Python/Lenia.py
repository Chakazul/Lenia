import numpy as np
import scipy.ndimage as snd
from fractions import Fraction
import copy
import re
import itertools
import datetime
import os
import json
import cv2
from PIL import Image, ImageTk
try:
	import tkinter as tk
except ImportError:
	import Tkinter as tk

SIZEX, SIZEY = 1 << 8, 1 << 8
# SIZEX, SIZEY = 1920, 1080  # 1080p HD
# SIZEX, SIZEY = 1280, 720  # 720p HD
MIDX, MIDY = int(SIZEX / 2), int(SIZEY / 2)
BORDER = 0  # 10
DEF_ZOOM = 4

RECORD_ROOT = 'record'
FRAME_EXT = '.png'
VIDEO_EXT = '.mp4'
VIDEO_CODEC = 'mp4v'  # .avi *'DIVX' / .mp4 *'mp4v' *'avc1'

class Board:
	def __init__(self, size=[0,0]):
		self.names = ['', '', '']
		self.params = {'R':11, 'T':10, 'b':[1], 'm':0.15, 's':0.015, 'kn':0, 'gn':0}
		self.cells = np.zeros(size)
		self.shift = [0, 0]
		self.rotate = 0

	def from_values(self, names, params, cells):
		self.names = names.copy()
		self.params = params.copy()
		self.cells = cells.copy()

	def from_data(self, data):
		self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
		self.params = data.get('params')
		if self.params is not None:
			self.params = self.params.copy()
			self.params['b'] = Board.st2fracs(self.params['b'])
		self.cells = data.get('cells')
		if self.cells is not None:
			self.cells = Board.rle2arr(self.cells)
		self.shift = [0, 0]
		self.rotate = 0

	def to_data(self, is_shorten=True):
		rle_st = Board.arr2rle(self.cells, is_shorten)
		params2 = self.params.copy()
		params2['b'] = Board.fracs2st(self.params['b'])
		data = {'code':self.names[0], 'name':self.names[1], 'cname':self.names[2], 'params':params2, 'cells':rle_st}
		return data

	def longname(self):
		return ' | '.join(filter(None, self.names))

	@staticmethod
	def arr2rle(A, is_shorten=True):
		''' http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
			http://golly.sourceforge.net/Help/formats.html#rle
			https://www.rosettacode.org/wiki/Run-length_encoding#Python
			A=[0 0 1] <-> V=[0 0 255] <-> code=[. . yO] <-> rle=[(2 .)(1 yO)] <-> st='2.yO'
			0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO '''
		V = np.rint(A*255).astype(int).tolist()
		code = [ [' .' if v==0 else ' '+chr(ord('A')+v-1) if v<25 else chr(ord('p')+(v-25)//24) + chr(ord('A')+(v-25)%24) for v in row] for row in V]
		if is_shorten:
			rle = [ [(len(list(g)),k.strip()) for k,g in itertools.groupby(row)] for row in code]
			for row in rle:
				if row[-1][1]=='.': row.pop()
			st = '$'.join(''.join([(str(n) if n>1 else '')+k for n,k in row]) for row in rle) + '!'
		else:
			st = '$'.join(''.join(row) for row in code) + '!'
		# print(sum(sum(r) for r in V))
		return st

	@staticmethod
	def rle2arr(st):
		lines = st.rstrip('!').split('$')
		rle = [re.findall('(\d*)([p-y]?[.boA-X])', row) for row in lines]
		code = [ sum([[k] * (1 if n=='' else int(n)) for n,k in row], []) for row in rle]
		V = [ [0 if c=='.' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row ] for row in code]
		maxlen = len(max(V, key=len))
		A = np.array([row + [0] * (maxlen - len(row)) for row in V])/255
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

	def add(self, board2, shift=[0,0], is_random=False):
		h1, w1 = self.cells.shape
		h2, w2 = min(board2.shape, self.cells.shape)
		if is_random:
			i = np.random.randint(w1 + w2) - w2//2
			j = np.random.randint(h1 + h2) - h2//2
		else:
			i = (w1 - w2)//2 - shift[1]
			j = (h1 - h2)//2 - shift[0]
		# self.cells[j:j+h2, i:i+w2] = board2[0:h2, 0:w2]
		for y in range(h2):
			for x in range(w2):
				if board2[y, x] > 0:
					self.cells[(j+y)%h1, (i+x)%w1] = board2[y, x]

	def transform(self, angle, new_R, is_replace=False):
		A = snd.rotate(self.cells, angle, order=0)
		A = snd.zoom(A, new_R / self.params['R'], order=0)
		if is_replace: self.cells = A
		return A

	def crop(self, min=1/255, is_replace=False):
		coords = np.argwhere(self.cells >= min)
		y0, x0 = coords.min(axis=0)
		y1, x1 = coords.max(axis=0) + 1
		A = self.cells[y0:y1, x0:x1]
		if is_replace: self.cells = A
		return A

class Recorder:
	def __init__(self):
		self.is_recording = False
		self.is_save_frames = False
		self.record_ID = None
		self.record_seq = None

	def toggle_recording(self, is_save_frames=False):
		self.is_save_frames = is_save_frames
		if not self.is_recording:
			self.start_record()
		else:
			self.finish_record()

	def start_record(self):
		''' https://github.com/cisco/openh264/ '''
		self.is_recording = True
		print('> start ' + ('saving frames and ' if self.is_save_frames else '') + 'recording video...')
		self.record_ID = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
		self.record_seq = 1
		if self.is_save_frames:
			self.img_dir = os.path.join(RECORD_ROOT, self.record_ID)
			if not os.path.exists(self.img_dir):
				os.makedirs(self.img_dir)
		self.video_path = os.path.join(RECORD_ROOT, '{}'.format(self.record_ID) + VIDEO_EXT)
		codec = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
		self.video = cv2.VideoWriter(self.video_path, codec, 24, (SIZEX,SIZEY))

	def record_frame(self, img):
		if self.is_save_frames:
			img_path = os.path.join(RECORD_ROOT, self.record_ID, '{:03d}'.format(self.record_seq) + FRAME_EXT)
			img.save(img_path)
		else:
			img_rgb = np.array(img.convert('RGB'))
			img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
			self.video.write(img_bgr)
		self.record_seq += 1

	def finish_record(self):
		if self.is_save_frames:
			for img_name in sorted(os.listdir(self.img_dir)):
				if img_name.endswith(FRAME_EXT):
					self.video.write(cv2.imread(os.path.join(self.img_dir, img_name)))
			print('> frames saved to "' + self.img_dir + '/*' + FRAME_EXT + '"')
		self.video.release()
		print('> video  saved to "' + self.video_path + '"')
		self.is_recording = False

class Automaton:
	kernel_core = {
		0: lambda r: (4 * r * (1-r))**4,
		1: lambda r: np.exp( 4 - 1 / (r * (1-r)) )
	}
	field_func = {
		0: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,
		1: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1
	}

	def __init__(self, world):
		self.world = world
		self.potential = np.zeros(world.cells.shape)
		self.field = np.zeros(world.cells.shape)
		self.calc_kernel()

	def kernel_shell(self, r):
		k = len(self.world.params['b'])
		kr = k * r
		bs = np.array([float(f) for f in self.world.params['b']])
		b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
		kfunc = Automaton.kernel_core[self.world.params['kn']]
		return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b

	def calc_once(self):
		world_FFT = np.fft.fft2(self.world.cells)
		potential = np.real(np.fft.ifft2(self.kernel_FFT * world_FFT))
		self.potential = np.roll(potential, (MIDY, MIDX), (0, 1))
		gfunc = Automaton.field_func[self.world.params['gn']]
		self.field = gfunc(self.potential, self.world.params['m'], self.world.params['s'])
		self.world.cells = np.clip(self.world.cells + self.field / self.world.params['T'], 0, 1)

	def calc_kernel(self):
		I, J = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
		X = (I-MIDX) / self.world.params['R']
		Y = (J-MIDY) / self.world.params['R']
		self.D = np.sqrt(X**2 + Y**2)

		self.kernel = self.kernel_shell(self.D)
		self.kernel_sum = np.sum(self.kernel)
		kernel_norm = self.kernel / self.kernel_sum
		self.kernel_FFT = np.fft.fft2(kernel_norm)
		self.kernel_updated = False

class Lenia:
	def __init__(self):
		self.is_run = True
		self.is_once = False
		self.show_what = 1
		self.zoom = DEF_ZOOM
		self.jet = self.create_palette(256, np.array([[0,0,1],[0,0,2],[0,1,2],[0,2,2],[1,2,1],[2,2,0],[2,1,0],[2,0,0],[1,0,0]]))
		self.read_animals()
		self.world = Board((SIZEY, SIZEX))
		self.automaton = Automaton(self.world)
		self.create_window()
		self.recorder = Recorder()
		self.focus = self.world

	def key_press(self, event):
		k = event.keysym.lower()
		s = event.state
		if s & 0x20000: k = 'A+' + k
		if s & 0x4: k = 'C+' + k
		if s & 0x1: k = 'S+' + k
		linear_add = 1 if 'S+' in k else 10
		double_mul = 1 if 'S+' in k else 2
		double_add = 1 if 'S+' in k else 0
		if k in ['escape']: self.close()
		elif k in ['enter', 'return']: self.is_run = not self.is_run
		elif k in [' ', 'space']: self.is_once = not self.is_once; self.is_run = False
		elif k in ['tab']: self.show_what = self.show_what % 4 + 1
		elif k in ['backspace', 'delete']: self.world.clear()
		elif k in ['q']: self.world.params['m'] += 0.01
		elif k in ['a']: self.world.params['m'] -= 0.01
		elif k in ['w']: self.world.params['s'] += 0.001
		elif k in ['s']: self.world.params['s'] -= 0.001
		elif k in ['S+q']: self.world.params['m'] += 0.001
		elif k in ['S+a']: self.world.params['m'] -= 0.001
		elif k in ['S+w']: self.world.params['s'] += 0.0001
		elif k in ['S+s']: self.world.params['s'] -= 0.0001
		elif k in ['t']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] // 2))
		elif k in ['g']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] * 2))
		elif k in ['S+t']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] - 1))
		elif k in ['S+g']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] + 1))
		# elif k in ['r']: self.zoom = max(1, min(20, self.zoom + 1)); self.load_animal_ID(self.animal_ID)
		# elif k in ['f']: self.zoom = max(1, min(20, self.zoom - 1)); self.load_animal_ID(self.animal_ID)
		elif k in ['r', 'S+r']: self.world.params['R'] = max(1, min(max(SIZEX,SIZEY), int(self.world.params['R'] * double_mul + double_add))); self.transform_world()
		elif k in ['f', 'S+f']: self.world.params['R'] = max(1, min(max(SIZEX,SIZEY), int(self.world.params['R'] / double_mul - double_add))); self.transform_world()
		elif k in ['down',  'S+down']:  self.focus.shift[0] = self.focus.shift[0] - linear_add; self.transform_world()
		elif k in ['up',    'S+up']:    self.focus.shift[0] = self.focus.shift[0] + linear_add; self.transform_world()
		elif k in ['right', 'S+right']: self.focus.shift[1] = self.focus.shift[1] - linear_add; self.transform_world()
		elif k in ['left',  'S+left']:  self.focus.shift[1] = self.focus.shift[1] + linear_add; self.transform_world()
		elif k in ['prior', 'S+prior']: self.focus.rotate = self.focus.rotate + linear_add; self.transform_world()
		elif k in ['next',  'S+next']:  self.focus.rotate = self.focus.rotate - linear_add; self.transform_world()
		elif k in ['1']: self.load_preset_buffer(1)
		elif k in ['2']: self.load_preset_buffer(2)
		elif k in ['z']: self.load_animal_ID(self.animal_ID)
		elif k in ['x']: self.load_buffer(self.buffer, is_replace=False, is_random=True)
		elif k in ['c']: self.load_animal_ID(self.animal_ID - 1)
		elif k in ['v']: self.load_animal_ID(self.animal_ID + 1)
		elif k in ['S+c']: self.load_animal_ID(self.animal_ID - 10)
		elif k in ['S+v']: self.load_animal_ID(self.animal_ID + 10)
		elif k in ['C+c', 'S+C+c', 'C+s', 'S+C+s']:
			A = copy.deepcopy(self.world)
			A.crop(1/255, is_replace=True)
			data = A.to_data(is_shorten='S+' not in k)
			if k.endswith('c'):
				self.clipboard_st = json.dumps(data, separators=(',', ':'))
				self.win.clipboard_clear()
				self.win.clipboard_append(self.clipboard_st)
				# print(self.clipboard_st)
				print('> board saved to clipboard')
			elif k.endswith('s'):
				with open('last_animal.json', 'w') as file:
					json.dump(data, file, separators=(',', ':'))
				with open('last_animal.rle', 'w') as file:
					file.write('#N '+A.longname()+'\n')
					data['params']['b'] = '[' + data['params']['b'] + ']'
					params_st = ','.join(['{}={}'.format(k,str(v)) for k,v in data['params'].items()])
					file.write('x = '+str(A.cells.shape[1])+', y = '+str(A.cells.shape[0])+', rule = Lenia('+params_st+')\n')
					file.write(data['cells'].replace('$','$\n')+'\n')
				print('> board saved to files "{}" & "{}"'.format('last_animal.json', 'last_animal.rle'))
		elif k in ['C+v']:
			self.clipboard_st = self.win.clipboard_get()
			data = json.loads(self.clipboard_st)
			self.buffer = Board()
			self.buffer.from_data(data)
			self.focus = self.buffer
			self.load_buffer(self.buffer, zoom=1)
		elif k in ['C+r', 'S+C+r']: self.recorder.toggle_recording(is_save_frames='S+' in k)
		elif k.endswith('_l') or k.endswith('_r') or k in ['app']: pass
		else: print(k + '[' + hex(event.state) + ']')

	def read_animals(self):
		with open('animals.json', encoding='utf-8') as file:
			self.animal_data = json.load(file)
		self.animal_ID = 4

	def load_animal_ID(self, ID, **kwargs):
		self.animal_ID = max(0, min(len(self.animal_data)-1, ID))
		self.buffer = Board()
		self.buffer.from_data(self.animal_data[self.animal_ID])
		self.load_buffer(self.buffer, **kwargs)

	def load_preset_buffer(self, id, **kwargs):
		if id==1: names = ['', 'Orbium bicaudatus', '']; params = {'R':13,'T':10,'b':[Fraction(1)],'m':0.15,'s':0.014,'kn':0,'gn':0}; cells = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],[0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],[0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],[0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],[0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],[0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],[0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],[0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],[0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],[0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],[0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],[0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],[0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],[0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],[0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],[0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],[0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],[0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],[0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],[0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
		elif id==2: names = ['', 'Rotorbium', '']; params = {'R':13,'T':10,'b':[Fraction(1)],'m':0.156,'s':0.0224,'kn':0,'gn':0}; cells = np.array([[0,0,0,0,0,0,0,0,0.003978,0.016492,0.004714,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.045386,0.351517,0.417829,0.367137,0.37766,0.426948,0.431058,0.282864,0.081247,0,0,0,0,0,0],[0,0,0,0,0.325473,0.450995,0.121737,0,0,0,0.003113,0.224278,0.47101,0.456459,0.247231,0.071609,0.013126,0,0,0],[0,0,0,0.386337,0.454077,0,0,0,0,0,0,0,0.27848,0.524466,0.464281,0.242651,0.096721,0.038476,0,0],[0,0,0.258817,0.583802,0.150994,0,0,0,0,0,0,0,0.226639,0.548329,0.550422,0.334764,0.153108,0.087049,0.042872,0],[0,0.008021,0.502406,0.524042,0.059531,0,0,0,0,0,0,0.033946,0.378866,0.615467,0.577527,0.357306,0.152872,0.090425,0.058275,0.023345],[0,0.179756,0.596317,0.533619,0.162612,0,0,0,0,0.015021,0.107673,0.325125,0.594765,0.682434,0.594688,0.381172,0.152078,0.073544,0.054424,0.030592],[0,0.266078,0.614339,0.605474,0.379255,0.195176,0.16516,0.179148,0.204498,0.299535,0.760743,1,1,1,1,0.490799,0.237826,0.069989,0.043549,0.022165],[0,0.333031,0.64057,0.686886,0.60698,0.509866,0.450525,0.389552,0.434978,0.859115,0.94097,1,1,1,1,1,0.747866,0.118317,0.037712,0.006271],[0,0.417887,0.6856,0.805342,0.824229,0.771553,0.69251,0.614328,0.651704,0.843665,0.910114,1,1,0.81765,0.703404,0.858469,1,0.613961,0.035691,0],[0.04674,0.526827,0.787644,0.895984,0.734214,0.661746,0.670024,0.646184,0.69904,0.723163,0.682438,0.618645,0.589858,0.374017,0.30658,0.404027,0.746403,0.852551,0.031459,0],[0.130727,0.658494,0.899652,0.508352,0.065875,0.009245,0.232702,0.419661,0.461988,0.470213,0.390198,0.007773,0,0.010182,0.080666,0.17231,0.44588,0.819878,0.034815,0],[0.198532,0.810417,0.63725,0.031385,0,0,0,0,0.315842,0.319248,0.321024,0,0,0,0,0.021482,0.27315,0.747039,0,0],[0.217619,0.968727,0.104843,0,0,0,0,0,0.152033,0.158413,0.114036,0,0,0,0,0,0.224751,0.647423,0,0],[0.138866,1,0.093672,0,0,0,0,0,0.000052,0.015966,0,0,0,0,0,0,0.281471,0.455713,0,0],[0,1,0.145606,0.005319,0,0,0,0,0,0,0,0,0,0,0,0.016878,0.381439,0.173336,0,0],[0,0.97421,0.262735,0.096478,0,0,0,0,0,0,0,0,0,0,0.013827,0.217967,0.287352,0,0,0],[0,0.593133,0.2981,0.251901,0.167326,0.088798,0.041468,0.013086,0.002207,0.009404,0.032743,0.061718,0.102995,0.1595,0.24721,0.233961,0.002389,0,0,0],[0,0,0.610166,0.15545,0.200204,0.228209,0.241863,0.243451,0.270572,0.446258,0.376504,0.174319,0.154149,0.12061,0.074709,0,0,0,0,0],[0,0,0.354313,0.32245,0,0,0,0.151173,0.479517,0.650744,0.392183,0,0,0,0,0,0,0,0,0],[0,0,0,0.329339,0.328926,0.176186,0.198788,0.335721,0.534118,0.549606,0.361315,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.090407,0.217992,0.190592,0.174636,0.222482,0.375871,0.265924,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.050256,0.235176,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.180145,0.132616,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.092581,0.188519,0.118256,0,0,0,0]]) 
		self.buffer = Board()
		self.buffer.from_values(names, params, cells)
		self.load_buffer(self.buffer, **kwargs)

	def load_buffer(self, board, is_replace=True, is_random=False, zoom=None):
		if zoom == None:
			zoom = self.zoom
		if board.names[0].startswith('~'):
			board.names[0] = board.names[0].lstrip('~')
			zoom = 1
		if is_replace:
			self.world.names = board.names.copy()
		if board.params is not None and board.cells is not None:
			if is_replace:
				self.world.params = board.params.copy()
				self.world.params['R'] *= zoom
				self.automaton.calc_kernel()
				self.world.clear()
			self.world.shift = [0, 0]
			self.world.rotate = np.random.random() * 360 if is_random else 0
			self.world.add(board.transform(self.world.rotate, self.world.params['R']), self.world.shift, is_random=is_random)
		self.update_title(zoom=zoom)

	def transform_world(self):
		self.world.clear()
		self.automaton.calc_kernel()
		self.world.add(self.buffer.transform(self.world.rotate, self.world.params['R']), self.world.shift)

	def update_title(self, zoom):
		if zoom == None:
			zoom = self.zoom
		st = ['Lenia ']
		st.append('['+self.world.longname()+']')
		st.append('x'+str(zoom))
		if self.recorder.is_recording:
			st.append(' '+chr(0x2B24)+'REC')
		self.win.title(' '.join(st))

	def create_window(self):
		# fig_size = SIZE * 2 + BORDER * 3
		fig_sizeX = SIZEX + BORDER * 2
		fig_sizeY = SIZEY + BORDER * 2
		self.win = tk.Tk()
		self.win.title('Lenia')
		self.win.bind("<Key>", self.key_press)
		self.frame = tk.Frame(self.win, width=fig_sizeX, height=fig_sizeY)
		self.frame.pack()
		self.canvas = tk.Canvas(self.frame, width=fig_sizeX, height=fig_sizeY)
		self.canvas.place(x=-1, y=-1)
		self.panel1 = self.create_panel(0, 0)
		# self.panel2 = self.create_panel(1, 0)
		# self.panel3 = self.create_panel(0, 1)
		# self.panel4 = self.create_panel(1, 1)

	def create_panel(self, c, r):
		buffer = np.uint8(np.zeros((SIZEY,SIZEX)))
		img = Image.frombuffer('P', (SIZEX,SIZEY), buffer, 'raw', 'P', 0, 1)
		photo = ImageTk.PhotoImage(image=img)
		return self.canvas.create_image(c*SIZEY+(c+1)*BORDER, r*SIZEX+(r+1)*BORDER, image=photo, anchor=tk.NW)

	def create_palette(self, nval, colors):
		ncol = colors.shape[0]
		colors = np.vstack(( colors, np.array([0,0,0]) ))
		v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 255 255 255]
		i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
		k = v / (nval-1) * (ncol-1)  # interpolate between 0 .. ncol-1
		k1 = k.astype(int)
		c1, c2 = colors[k1,i], colors[k1+1,i]
		c = (k-k1) * (c2-c1) + c1  # interpolate between c1 .. c2
		return np.rint(c / 2 * 255).astype(int).tolist()

	def show_world(self):
		if self.show_what == 1: self.show_panel(self.panel1, self.world.cells, 0, 1)
		elif self.show_what == 2: self.show_panel(self.panel1, self.automaton.potential, 0, 0.3)
		elif self.show_what == 3: self.show_panel(self.panel1, self.automaton.field, -1, 1)
		elif self.show_what == 4: self.show_panel(self.panel1, self.automaton.kernel, 0, 1)
		# if not self.kernel_updated:
			# self.show_panel(self.panel4, self.kernel, 0, 1)
			# self.kernel_updated = True
		#self.win.update()

	def show_panel(self, panel, A, vmin=0, vmax=1):
		buffer = np.uint8((A-vmin) / (vmax-vmin) * 255).copy(order='C')
		img = Image.frombuffer('P', (SIZEX,SIZEY), buffer, 'raw', 'P', 0, 1)
		img.putpalette(self.jet)
		if self.recorder.is_recording:
			self.recorder.record_frame(img)
		photo = ImageTk.PhotoImage(image=img)
		# photo = tk.PhotoImage(width=SIZEX, height=SIZEY)
		self.canvas.itemconfig(panel, image=photo)
		self.win.update()

	def loop(self):
		self.is_loop = True
		self.win.after(0, self.run)
		self.win.protocol("WM_DELETE_WINDOW", self.close)
		self.win.mainloop()

	def close(self):
		self.is_loop = False
		if self.recorder.is_recording:
			self.recorder.finish_record()
		self.win.destroy()

	def run(self):
		while self.is_loop:
			if self.is_run or self.is_once:
				self.automaton.calc_once()
				self.is_once = False
			self.show_world()

if __name__ == '__main__':
	lenia = Lenia()
	lenia.load_preset_buffer(1, zoom=8)
	lenia.loop()

# GPU FFT
# https://pythonhosted.org/pyfft/
# http://arrayfire.org/docs/group__signal__func__fft2.htm
