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
DEF_ZOOM = 4
EPSILON = 1e-10

RECORD_ROOT = 'record'
FRAME_EXT = '.png'
VIDEO_EXT = '.mp4'
VIDEO_CODEC = 'mp4v'  # .avi *'DIVX' / .mp4 *'mp4v' *'avc1'

class Board:
	def __init__(self, size=[0,0]):
		self.names = ['', '', '']
		self.params = {'R':11, 'T':10, 'b':[1], 'm':0.15, 's':0.015, 'kn':0, 'gn':0}
		self.cells = np.zeros(size)

	@classmethod
	def from_values(cls, names, params, cells):
		self = cls()
		self.names = names.copy()
		self.params = params.copy()
		self.cells = cells.copy()
		return self

	@classmethod
	def from_data(cls, data):
		self = cls()
		self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
		self.params = data.get('params')
		if self.params is not None:
			self.params = self.params.copy()
			self.params['b'] = Board.st2fracs(self.params['b'])
		self.cells = data.get('cells')
		if self.cells is not None:
			self.cells = Board.rle2arr(self.cells)
		return self

	def to_data(self, is_shorten=True):
		rle_st = Board.arr2rle(self.cells, is_shorten)
		params2 = self.params.copy()
		params2['b'] = Board.fracs2st(self.params['b'])
		data = {'code':self.names[0], 'name':self.names[1], 'cname':self.names[2], 'params':params2, 'cells':rle_st}
		return data

	def params2st(self):
		params2 = self.params.copy()
		params2['b'] = '[' + Board.fracs2st(self.params['b']) + ']'
		return ','.join(['{}={}'.format(k,str(v)) for k,v in params.items()])

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

	def add(self, part, shift=[0,0]):
		# assert self.params['R'] == part.params['R']
		h1, w1 = self.cells.shape
		h2, w2 = part.cells.shape
		h, w = min(part.cells.shape, self.cells.shape)
		i1 = (w1 - w)//2 + shift[1]
		j1 = (h1 - h)//2 + shift[0]
		i2 = (w2 - w)//2
		j2 = (h2 - h)//2
		# self.cells[j:j+h, i:i+w] = part.cells[0:h, 0:w]
		for y in range(h):
			for x in range(w):
				if part.cells[j2+y, i2+x] > 0:
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
		if 'S' in mode and tx['shift'] != [0, 0]:
			self.cells = snd.shift(self.cells, tx['shift'], order=0, mode='wrap')
			# self.cells = np.roll(self.cells, tx['shift'], (0, 1))
		if 'F' in mode and tx['flip'] != -1:
			self.cells = np.flip(self.cells, tx['flip'])
		return self

	def add_transformed(self, part, tx):
		part = copy.deepcopy(part)
		self.add(part.transform(tx, mode='RZF'), tx['shift'])

	def crop(self, min=1/255):
		coords = np.argwhere(self.cells >= min)
		y0, x0 = coords.min(axis=0)
		y1, x1 = coords.max(axis=0) + 1
		self.cells = self.cells[y0:y1, x0:x1]
		return self

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
		self.field_old = None
		self.calc_kernel()
		self.gen = 0
		self.time = 0
		self.is_multistep = False

	def kernel_shell(self, r):
		k = len(self.world.params['b'])
		kr = k * r
		bs = np.array([float(f) for f in self.world.params['b']])
		b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
		kfunc = Automaton.kernel_core[self.world.params['kn']]
		return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b

	def calc_once(self):
		world_FFT = np.fft.fft2(self.world.cells)
		potential_shifted = np.real(np.fft.ifft2(self.kernel_FFT * world_FFT))
		self.potential = np.roll(potential_shifted, (MIDY, MIDX), (0, 1))
		gfunc = Automaton.field_func[self.world.params['gn']]
		self.field = gfunc(self.potential, self.world.params['m'], self.world.params['s'])
		dt = 1 / self.world.params['T']
		if self.is_multistep and self.field_old is not None:
			self.world.cells = np.clip(self.world.cells + 1/2 * dt * (3 * self.field - self.field_old), 0, 1)
		else:
			self.world.cells = np.clip(self.world.cells + dt * self.field, 0, 1)
		if self.is_multistep:
			self.field_old = self.field.copy()
		self.gen += 1
		self.time += dt

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

	def reset(self):
		self.gen = 0
		self.time = 0
		self.field_old = None

class Lenia:
	def __init__(self):
		self.is_run = True
		self.is_once = False
		self.fore = None
		self.back = None
		self.is_composit = False
		self.is_auto_center = False
		self.show_what = 0
		self.zoom = DEF_ZOOM
		self.colormap = [
			self.create_colormap(256, np.array([[0,0,0,0,4,8,8,8,4],[0,0,4,8,8,8,4,0,0],[4,8,8,8,4,0,0,0,0]])), #BCYR
			self.create_colormap(256, np.array([[0,0,4,8,8,8,4,0,0],[2,4,6,8,4,0,0,0,0],[0,0,0,0,4,8,8,8,4]])), #GYPB
			self.create_colormap(256, np.array([[4,8,8,8,4,0,0,0,0],[0,0,4,8,8,8,6,4,2],[0,0,0,0,0,0,0,0,0]])), #RYGG
			self.create_colormap(256, np.array([[0,3,4,5,8],[0,3,4,5,8],[0,3,4,5,8]]))] #B/W
		self.colormap_ID = 0

		self.read_animals()
		self.world = Board((SIZEY, SIZEX))
		self.automaton = Automaton(self.world)
		self.clear_transform()
		self.create_window()
		self.recorder = Recorder()

	def clear_transform(self):
		self.tx = {'shift':[0, 0], 'rotate':0, 'R':self.world.params['R'], 'flip':-1}

	def key_press(self, event):
		# Win: shift_l/r(0x1) caps_lock(0x2) control_l/r(0x4) alt_l/r(0x20000) win/app/alt_r/control_r(0x40000)
		# Mac: shift_l(0x1) caps_lock(0x2) control_l(0x4) command_l(0x8)
		k = event.keysym.lower()
		s = event.state
		# print(k + '[' + hex(event.state) + ']'); return
		if s & 0x20000: k = 'A+' + k  # Win: Alt
		#if s & 0x8: k = 'C+' + k  # Mac: Command
		if s & 0x4: k = 'C+' + k  # Win/Mac: Control
		if s & 0x1: k = 'S+' + k
		shift1 = 1 if 'S+' not in k else -1
		add10 = 10 if 'S+' not in k else 1
		add1 = 1 if 'S+' not in k else 10
		mul1 = 1 if 'S+' not in k else 0
		mul2 = 2 if 'S+' not in k else 1
		add2 = 0 if 'S+' not in k else 1
		if k in ['escape']: self.close()
		elif k in ['enter', 'return']: self.is_run = not self.is_run
		elif k in [' ', 'space']: self.is_once = not self.is_once; self.is_run = False
		elif k in ['tab', 'S+tab']: self.show_what = (self.show_what + shift1) % 4
		elif k in ['backspace', 'delete']: self.world.clear(); self.automaton.reset()
		elif k in ['q', 'S+q']: self.world.params['m'] += add10 * 0.001
		elif k in ['a', 'S+a']: self.world.params['m'] -= add10 * 0.001
		elif k in ['w', 'S+w']: self.world.params['s'] += add10 * 0.0001
		elif k in ['s', 'S+s']: self.world.params['s'] -= add10 * 0.0001
		elif k in ['t', 'S+t']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] // mul2 - add2))
		elif k in ['g', 'S+g']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] *  mul2 + add2))
		elif k in ['C+o']: self.automaton.is_multistep = not self.automaton.is_multistep
		elif k in ['C+p']: self.world.params['T'] *= -1; self.world.params['m'] = 1 - self.world.params['m']; self.world.cells = 1 - self.world.cells
		elif k in ['r', 'S+r']: self.set_zoom(+mul1, +add2); self.transform_world()
		elif k in ['f', 'S+f']: self.set_zoom(-mul1, -add2); self.transform_world()
		elif k in ['down',  'S+down']:  self.tx['shift'][0] += add10; self.transform_world()
		elif k in ['up',    'S+up']:    self.tx['shift'][0] -= add10; self.transform_world()
		elif k in ['right', 'S+right']: self.tx['shift'][1] += add10; self.transform_world()
		elif k in ['left',  'S+left']:  self.tx['shift'][1] -= add10; self.transform_world()
		elif k in ['prior', 'S+prior']: self.tx['rotate'] += add10; self.transform_world()
		elif k in ['next',  'S+next']:  self.tx['rotate'] -= add10; self.transform_world()
		elif k in ['home']: self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
		elif k in ['end']:  self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
		elif k in ['1']: self.load_preset_part(1)
		elif k in ['2']: self.load_preset_part(2)
		elif k in ['z', 'S+z']: self.load_animal_ID(self.animal_ID, mode='Composit' if 'S+' in k else 'Replace')
		elif k in ['x', 'S+x']: self.load_part(self.fore, is_random=True, mode='Composit' if 'S+' in k else 'Add')
		elif k in ['c', 'S+c']: self.load_animal_ID(self.animal_ID - add1)
		elif k in ['v', 'S+v']: self.load_animal_ID(self.animal_ID + add1)
		elif k in ['m']: self.center_world()
		elif k in ['C+m']: self.center_world(); self.is_auto_center = not self.is_auto_center
		elif k in ['C+c', 'S+C+c', 'C+s', 'S+C+s']:
			A = copy.deepcopy(self.world)
			A.crop(1/255)
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
					file.write('x = '+str(A.cells.shape[1])+', y = '+str(A.cells.shape[0])+', rule = Lenia('+A.params2st()+')\n')
					file.write(data['cells'].replace('$','$\n')+'\n')
				print('> board saved to files "{}" & "{}"'.format('last_animal.json', 'last_animal.rle'))
		elif k in ['C+v', 'S+C+v']:
			self.clipboard_st = self.win.clipboard_get()
			data = json.loads(self.clipboard_st)
			self.load_part(Board.from_data(data), zoom=1, mode='Composit' if 'S+' in k else 'Replace')
		elif k in ['quoteleft']: self.is_composit = False
		elif k in ['C+r', 'S+C+r']: self.recorder.toggle_recording(is_save_frames='S+' in k)
		elif k in ['equal']: self.colormap_ID = (self.colormap_ID + 1) % len(self.colormap)
		elif k.endswith('_l') or k.endswith('_r'): pass
		else: print(k + '[' + hex(event.state) + ']')
		self.world.params['m'] = round(self.world.params['m'], 10)
		self.world.params['s'] = round(self.world.params['s'], 10)
		self.update_info()

	def clear_screen(self):
		_ = os.system('cls' if os.name == 'nt' else 'clear')

	def update_info(self):
		# self.clear_screen()
		# st = "Lenia  animal[{name}]({w}{times}{h})".format(name=self.world.longname(), h=self.fore.cells.shape[0], w=self.fore.cells.shape[1], times=chr(0xD7))
		st = "Lenia  [{name}] {times}{zoom}".format(name=self.world.longname(), zoom=self.zoom, times=chr(0xD7))
		if self.recorder.is_recording:
			st += "  "+chr(0x2B24)+"REC"
		# print(st)
		self.win.title(st)

	def read_animals(self):
		with open('animals.json', encoding='utf-8') as file:
			self.animal_data = json.load(file)
		self.animal_ID = 4

	def load_animal_ID(self, ID, **kwargs):
		self.animal_ID = max(0, min(len(self.animal_data)-1, ID))
		self.load_part(Board.from_data(self.animal_data[self.animal_ID]), **kwargs)

	def load_preset_part(self, id, **kwargs):
		if id==1: names = ['', 'Orbium bicaudatus', '']; params = {'R':13,'T':10,'b':[Fraction(1)],'m':0.15,'s':0.014,'kn':0,'gn':0}; cells = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],[0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],[0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],[0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],[0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],[0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],[0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],[0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],[0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],[0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],[0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],[0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],[0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],[0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],[0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],[0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],[0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],[0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],[0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],[0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
		elif id==2: names = ['', 'Rotorbium', '']; params = {'R':13,'T':10,'b':[Fraction(1)],'m':0.156,'s':0.0224,'kn':0,'gn':0}; cells = np.array([[0,0,0,0,0,0,0,0,0.003978,0.016492,0.004714,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.045386,0.351517,0.417829,0.367137,0.37766,0.426948,0.431058,0.282864,0.081247,0,0,0,0,0,0],[0,0,0,0,0.325473,0.450995,0.121737,0,0,0,0.003113,0.224278,0.47101,0.456459,0.247231,0.071609,0.013126,0,0,0],[0,0,0,0.386337,0.454077,0,0,0,0,0,0,0,0.27848,0.524466,0.464281,0.242651,0.096721,0.038476,0,0],[0,0,0.258817,0.583802,0.150994,0,0,0,0,0,0,0,0.226639,0.548329,0.550422,0.334764,0.153108,0.087049,0.042872,0],[0,0.008021,0.502406,0.524042,0.059531,0,0,0,0,0,0,0.033946,0.378866,0.615467,0.577527,0.357306,0.152872,0.090425,0.058275,0.023345],[0,0.179756,0.596317,0.533619,0.162612,0,0,0,0,0.015021,0.107673,0.325125,0.594765,0.682434,0.594688,0.381172,0.152078,0.073544,0.054424,0.030592],[0,0.266078,0.614339,0.605474,0.379255,0.195176,0.16516,0.179148,0.204498,0.299535,0.760743,1,1,1,1,0.490799,0.237826,0.069989,0.043549,0.022165],[0,0.333031,0.64057,0.686886,0.60698,0.509866,0.450525,0.389552,0.434978,0.859115,0.94097,1,1,1,1,1,0.747866,0.118317,0.037712,0.006271],[0,0.417887,0.6856,0.805342,0.824229,0.771553,0.69251,0.614328,0.651704,0.843665,0.910114,1,1,0.81765,0.703404,0.858469,1,0.613961,0.035691,0],[0.04674,0.526827,0.787644,0.895984,0.734214,0.661746,0.670024,0.646184,0.69904,0.723163,0.682438,0.618645,0.589858,0.374017,0.30658,0.404027,0.746403,0.852551,0.031459,0],[0.130727,0.658494,0.899652,0.508352,0.065875,0.009245,0.232702,0.419661,0.461988,0.470213,0.390198,0.007773,0,0.010182,0.080666,0.17231,0.44588,0.819878,0.034815,0],[0.198532,0.810417,0.63725,0.031385,0,0,0,0,0.315842,0.319248,0.321024,0,0,0,0,0.021482,0.27315,0.747039,0,0],[0.217619,0.968727,0.104843,0,0,0,0,0,0.152033,0.158413,0.114036,0,0,0,0,0,0.224751,0.647423,0,0],[0.138866,1,0.093672,0,0,0,0,0,0.000052,0.015966,0,0,0,0,0,0,0.281471,0.455713,0,0],[0,1,0.145606,0.005319,0,0,0,0,0,0,0,0,0,0,0,0.016878,0.381439,0.173336,0,0],[0,0.97421,0.262735,0.096478,0,0,0,0,0,0,0,0,0,0,0.013827,0.217967,0.287352,0,0,0],[0,0.593133,0.2981,0.251901,0.167326,0.088798,0.041468,0.013086,0.002207,0.009404,0.032743,0.061718,0.102995,0.1595,0.24721,0.233961,0.002389,0,0,0],[0,0,0.610166,0.15545,0.200204,0.228209,0.241863,0.243451,0.270572,0.446258,0.376504,0.174319,0.154149,0.12061,0.074709,0,0,0,0,0],[0,0,0.354313,0.32245,0,0,0,0.151173,0.479517,0.650744,0.392183,0,0,0,0,0,0,0,0,0],[0,0,0,0.329339,0.328926,0.176186,0.198788,0.335721,0.534118,0.549606,0.361315,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.090407,0.217992,0.190592,0.174636,0.222482,0.375871,0.265924,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.050256,0.235176,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.180145,0.132616,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.092581,0.188519,0.118256,0,0,0,0]]) 
		self.load_part(Board.from_values(names, params, cells), **kwargs)

	def load_part(self, part, mode='Replace', is_random=False, zoom=None):
		self.fore = part
		if zoom == None:
			zoom = self.zoom
		if part.names[0].startswith('~'):
			part.names[0] = part.names[0].lstrip('~')
			zoom = 1
		if mode=='Replace':
			self.world.names = part.names.copy()
		if part.params is not None and part.cells is not None:
			self.is_composit = False
			if mode=='Composit':
				self.back = copy.deepcopy(self.world)
				self.is_composit = True
			elif mode=='Replace':
				self.world.params = part.params.copy()
				self.world.params['R'] *= zoom
				self.automaton.calc_kernel()
				self.world.clear()
				self.automaton.reset()
			self.clear_transform()
			if is_random:
				self.tx['rotate'] = np.random.random() * 360
				h1, w1 = self.world.cells.shape
				h, w = min(part.cells.shape, self.world.cells.shape)
				self.tx['shift'][1] = np.random.randint(w1 + w) - w1//2
				self.tx['shift'][0] = np.random.randint(h1 + h) - h1//2
				self.tx['flip'] = np.random.randint(3) - 1
			self.world.add_transformed(part, self.tx)

	def set_zoom(self, zoom_add, R_add):
		if zoom_add != 0:
			zoom_old = self.zoom
			self.zoom = max(1, self.zoom + zoom_add)
			self.tx['R'] = self.tx['R'] // zoom_old * self.zoom
		self.tx['R'] += R_add

	def transform_world(self):
		if self.is_composit:
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
		self.world.params['R'] = self.tx['R']
		self.automaton.calc_kernel()

	def center_world(self):
		if np.sum(self.world.cells) < EPSILON:
			return
		cy, cx = snd.center_of_mass(self.world.cells)
		tx = {'shift':[MIDY - cy, MIDX - cx]}
		self.world.transform(tx, mode='S', is_world=True)

	def create_window(self):
		self.win = tk.Tk()
		self.win.title('Lenia')
		self.win.bind('<Key>', self.key_press)
		self.frame = tk.Frame(self.win, width=SIZEX, height=SIZEY)
		self.frame.pack()
		self.canvas = tk.Canvas(self.frame, width=SIZEX, height=SIZEY)
		self.canvas.place(x=-1, y=-1)
		self.panel1 = self.create_panel(0, 0)
		# self.panel2 = self.create_panel(1, 0)
		# self.panel3 = self.create_panel(0, 1)
		# self.panel4 = self.create_panel(1, 1)

	def create_panel(self, c, r):
		buffer = np.uint8(np.zeros((SIZEY,SIZEX)))
		img = Image.frombuffer('P', (SIZEX,SIZEY), buffer, 'raw', 'P', 0, 1)
		photo = ImageTk.PhotoImage(image=img)
		return self.canvas.create_image(c*SIZEY, r*SIZEX, image=photo, anchor=tk.NW)

	def create_colormap(self, nval, colors):
		ncol = colors.shape[1]
		colors = np.hstack((colors, np.array([[0],[0],[0]])))
		v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 255 255 255]
		i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
		k = v / (nval-1) * (ncol-1)  # interpolate between 0 .. ncol-1
		k1 = k.astype(int)
		c1, c2 = colors[i,k1], colors[i,k1+1]
		c = (k-k1) * (c2-c1) + c1  # interpolate between c1 .. c2
		return np.rint(c / 8 * 255).astype(int).tolist()

	def show_world(self):
		if self.show_what == 0: self.show_panel(self.panel1, self.world.cells, 0, 1)
		elif self.show_what == 1: self.show_panel(self.panel1, self.automaton.potential, 0, 0.3)
		elif self.show_what == 2: self.show_panel(self.panel1, self.automaton.field, -1, 1)
		elif self.show_what == 3: self.show_panel(self.panel1, self.automaton.kernel, 0, 1)
		# if not self.kernel_updated:
			# self.show_panel(self.panel4, self.kernel, 0, 1)
			# self.kernel_updated = True
		#self.win.update()

	def show_panel(self, panel, A, vmin=0, vmax=1):
		buffer = np.uint8((A-vmin) / (vmax-vmin) * 255).copy(order='C')
		img = Image.frombuffer('P', (SIZEX,SIZEY), buffer, 'raw', 'P', 0, 1)
		img.putpalette(self.colormap[self.colormap_ID])
		if self.recorder.is_recording:
			self.recorder.record_frame(img)
		photo = ImageTk.PhotoImage(image=img)
		# photo = tk.PhotoImage(width=SIZEX, height=SIZEY)
		self.canvas.itemconfig(panel, image=photo)
		self.win.update()

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
			if self.is_run or self.is_once:
				self.automaton.calc_once()
				if self.is_auto_center:
					self.center_world()
				if not self.is_composit:
					self.back = None
					self.clear_transform()
				self.is_once = False
			self.show_world()

if __name__ == '__main__':
	lenia = Lenia()
	lenia.load_preset_part(1, zoom=8)
	lenia.update_info()
	lenia.loop()

# GPU FFT
# https://pythonhosted.org/pyfft/
# http://arrayfire.org/docs/group__signal__func__fft2.htm
