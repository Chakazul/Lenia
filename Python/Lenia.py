''''
pip3 install numpy
pip3 install scipy
pip3 install opencv-python
pip3 install pillow
'''
import numpy as np
import scipy.ndimage as snd
from fractions import Fraction
import copy, re, itertools, json
import datetime, os, sys
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
DEF_ZOOM = 3
EPSILON = 1e-10
ROUND = 10

RECORD_ROOT = 'record'
FRAME_EXT = '.png'
VIDEO_EXT = '.mp4'
VIDEO_CODEC = 'mp4v'  # .avi *'DIVX' / .mp4 *'mp4v' *'avc1'

class Board:
	def __init__(self, size=[0,0]):
		self.names = ['', '', '']
		self.params = {'R':10, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
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
		params2['b'] = Board.fracs2st(params2['b'])
		data = {'code':self.names[0], 'name':self.names[1], 'cname':self.names[2], 'params':params2, 'cells':rle_st}
		return data

	def params2st(self):
		params2 = self.params.copy()
		params2['b'] = '[' + Board.fracs2st(params2['b']) + ']'
		return ','.join(['{}={}'.format(k,str(v)) for (k,v) in params2.items()])

	def long_name(self):
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
		return self

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
		self.record_id = None
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
		self.status = "> start " + ("saving frames and " if self.is_save_frames else "") + "recording video..."
		self.record_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
		self.record_seq = 1
		if self.is_save_frames:
			self.img_dir = os.path.join(RECORD_ROOT, self.record_id)
			if not os.path.exists(self.img_dir):
				os.makedirs(self.img_dir)
		self.video_path = os.path.join(RECORD_ROOT, '{}'.format(self.record_id) + VIDEO_EXT)
		codec = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
		self.video = cv2.VideoWriter(self.video_path, codec, 24, (SIZEX,SIZEY))

	def record_frame(self, img):
		if self.is_save_frames:
			img_path = os.path.join(RECORD_ROOT, self.record_id, '{:03d}'.format(self.record_seq) + FRAME_EXT)
			img.save(img_path)
		else:
			img_rgb = np.array(img.convert('RGB'))
			img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
			self.video.write(img_bgr)
		self.record_seq += 1

	def finish_record(self):
		status = []
		if self.is_save_frames:
			for img_name in sorted(os.listdir(self.img_dir)):
				if img_name.endswith(FRAME_EXT):
					self.video.write(cv2.imread(os.path.join(self.img_dir, img_name)))
			status.append("> frames saved to '" + self.img_dir + "/*" + FRAME_EXT + "'")
		self.video.release()
		status.append("> video  saved to '" + self.video_path + "'")
		self.status = "\n".join(status)
		self.is_recording = False

class Automaton:
	kernel_core = {
		0: lambda r: (4 * r * (1-r))**4,  # quad4
		1: lambda r: np.exp( 4 - 1 / (r * (1-r)) )  # bump4
	}
	field_func = {
		0: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # quad4
		1: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1  # gaus
	}

	def __init__(self, world):
		self.world = world
		self.potential = np.zeros(world.cells.shape)
		self.field = np.zeros(world.cells.shape)
		self.field_old = None
		self.change = np.zeros(world.cells.shape)
		self.calc_kernel()
		self.gen = 0
		self.time = 0
		self.is_multistep = False
		self.clip_mode = 0
		self.kn = 1
		self.gn = 1

	def kernel_shell(self, r):
		k = len(self.world.params['b'])
		kr = k * r
		bs = np.array([float(f) for f in self.world.params['b']])
		b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
		kfunc = Automaton.kernel_core[(self.world.params.get('kn') or self.kn) - 1]
		return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b

	@staticmethod
	def soft_max(x, m, k):
		''' https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
		return np.log(np.exp(k*x) + np.exp(k*m)) / k

	@staticmethod
	def soft_clip(x, min, max, k):
		a = np.exp(k*x)
		b = np.exp(k*min)
		c = np.exp(-k*max)
		return np.log( 1/(a+b)+c ) / -k
		# return Automaton.soft_max(Automaton.soft_max(x, min, k), max, -k)

	def calc_once(self, is_update=True):
		A = self.world.cells
		world_FFT = np.fft.fft2(A)
		potential_shifted = np.real(np.fft.ifft2(self.kernel_FFT * world_FFT))
		self.potential = np.roll(potential_shifted, (MIDY, MIDX), (0, 1))
		gfunc = Automaton.field_func[(self.world.params.get('gn') or self.gn) - 1]
		self.field = gfunc(self.potential, self.world.params['m'], self.world.params['s'])
		dt = 1 / self.world.params['T']
		if self.is_multistep and self.field_old is not None:
			D = 1/2 * (3 * self.field - self.field_old)
			self.field_old = self.field.copy()
		else:
			D = self.field
		if self.clip_mode==0:
			A_new = np.clip(A + dt * D, 0, 1)
			# A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
		elif self.clip_mode==1:
			A_new = Automaton.soft_clip(A + dt * D, 0, 1, 1/dt)
			# A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
		self.change = (A_new - A) / dt
		if is_update:
			self.world.cells = A_new
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
		self.is_layered = False
		self.is_auto_center = False
		self.is_auto_load = False
		self.show_what = 0
		self.zoom = DEF_ZOOM
		self.colormap = [
			self.create_colormap(256, np.array([[0,0,0,0,4,8,8,8,4],[0,0,4,8,8,8,4,0,0],[4,8,8,8,4,0,0,0,0]])), #BCYR
			self.create_colormap(256, np.array([[0,0,4,8,8,8,4,0,0],[2,4,6,8,4,0,0,0,0],[0,0,0,0,4,8,8,8,4]])), #GYPB
			self.create_colormap(256, np.array([[4,8,8,8,4,0,0,0,0],[0,0,4,8,8,8,6,4,2],[0,0,0,0,0,0,0,0,0]])), #RYGG
			self.create_colormap(256, np.array([[0,3,4,5,8],[0,3,4,5,8],[0,3,4,5,8]]))] #B/W
		self.colormap_id = 0
		self.status = ""

		self.read_animals()
		self.world = Board((SIZEY, SIZEX))
		self.automaton = Automaton(self.world)
		self.clear_transform()
		self.create_window()
		self.create_menu()
		self.recorder = Recorder()
		self.excess_key = None

	def clear_transform(self):
		self.tx = {'shift':[0, 0], 'rotate':0, 'R':self.world.params['R'], 'flip':-1}

	def check_auto_load(self):
		if self.is_auto_load:
			self.load_part(self.fore, is_set_params=False)

	def read_animals(self):
		with open('animals.json', encoding='utf-8') as file:
			self.animal_data = json.load(file)

	def load_animal_id(self, id, **kwargs):
		self.animal_id = max(0, min(len(self.animal_data)-1, id))
		self.load_part(Board.from_data(self.animal_data[self.animal_id]), **kwargs)

	def load_animal_code(self, code, **kwargs):
		id = self.get_animal_id(code)
		if id is not None:
			self.load_animal_id(id, **kwargs)

	def get_animal_id(self, code):
		code_sp = code.split(':')
		n = int(code_sp[1]) if len(code_sp)==2 else 1
		it = (id for (id, data) in enumerate(self.animal_data) if data["code"]==code_sp[0])
		for i in range(n):
			id = next(it, None)
		return id

	def load_part(self, part, is_replace=True, is_random=False, zoom=None, is_set_params=True):
		self.fore = part
		if zoom == None:
			zoom = self.zoom
		if part.names[0].startswith('~'):
			part.names[0] = part.names[0].lstrip('~')
			zoom = 1
		if is_replace:
			self.world.names = part.names.copy()
		if part.params is not None and part.cells is not None:
			if self.is_layered:
				self.back = copy.deepcopy(self.world)
			if is_replace and not self.is_layered:
				if is_set_params:
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
			self.tx['R'] = self.fore.params['R'] // zoom_old * self.zoom
		self.tx['R'] += R_add
		print(self.fore.params['R'])

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
		self.world.params['R'] = self.tx['R']
		self.automaton.calc_kernel()

	def center_world(self):
		if np.sum(self.world.cells) < EPSILON:
			return
		cy, cx = snd.center_of_mass(self.world.cells)
		tx = {'shift':[MIDY - cy, MIDX - cx]}
		self.world.transform(tx, mode='S', is_world=True)

	def clear_world(self):
		self.world.clear()
		if self.is_layered:
			self.back = copy.deepcopy(self.world)
		self.automaton.reset()

	def create_window(self):
		self.win = tk.Tk()
		self.win.title('Lenia')
		self.win.bind('<Key>', self.key_press_event)
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
		change_range = 1.4 if self.automaton.clip_mode>=1 else 1
		if self.show_what==0: self.show_panel(self.panel1, self.world.cells, 0, 1)
		elif self.show_what==1: self.show_panel(self.panel1, self.automaton.potential, 0, 1.9*self.world.params['m'])
		elif self.show_what==2: self.show_panel(self.panel1, self.automaton.field, -1, 1)
		elif self.show_what==3: self.show_panel(self.panel1, self.automaton.change, -change_range, change_range)
		elif self.show_what==4: self.show_panel(self.panel1, self.automaton.kernel, 0, 1)
		# if not self.kernel_updated:
			# self.show_panel(self.panel4, self.kernel, 0, 1)
			# self.kernel_updated = True
		#self.win.update()

	def show_panel(self, panel, A, vmin=0, vmax=1):
		buffer = np.uint8((A-vmin) / (vmax-vmin) * 255).copy(order='C')
		img = Image.frombuffer('P', (SIZEX,SIZEY), buffer, 'raw', 'P', 0, 1)
		img.putpalette(self.colormap[self.colormap_id])
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
				if not self.is_layered:
					self.back = None
					self.clear_transform()
				self.is_once = False
			self.show_world()

	def key_press_event(self, event):
		''' https://www.tcl.tk/man/tcl8.6/TkCmd/keysyms.htm '''
		# Win: shift_l/r(0x1) caps_lock(0x2) control_l/r(0x4) alt_l/r(0x20000) win/app/alt_r/control_r(0x40000)
		# Mac: shift_l(0x1) caps_lock(0x2) control_l(0x4) meta_l(0x8,command) alt_l(0x10) super_l(0x40,fn)
		# print('keysym[{0.keysym}] char[{0.char}] keycode[{0.keycode}] state[{1}]'.format(event, hex(event.state))); return
		k = event.keysym.lower()
		s = event.state
		if s & 0x20000: k = 'A+' + k  # Win: Alt
		#if s & 0x8: k = 'C+' + k  # Mac: Meta/Command
		if s & 0x4: k = 'C+' + k  # Win/Mac: Control
		if s & 0x1: k = 'S+' + k
		self.key_press(k)

	ANIMAL_KEY_LIST = {'1':'O2(a)', '2':'OG2', '3':'P4(a)', '4':'3G:4', '5':'4Q(5,5,5,5):3'}
	def key_press(self, k):
		inc_or_dec = 1 if 'S+' not in k else -1
		inc_10_or_1 = 10 if 'S+' not in k else 1
		inc_1_or_10 = 1 if 'S+' not in k else 10
		inc_mul_or_not = 1 if 'S+' not in k else 0
		double_or_not = 2 if 'S+' not in k else 1
		inc_or_not = 0 if 'S+' not in k else 1

		is_ignore = False
		self.excess_key = None
		self.status = ""
		if k in ['escape']: self.close()
		elif k in ['enter', 'return']: self.is_run = not self.is_run
		elif k in [' ', 'space']: self.is_once = not self.is_once; self.is_run = False
		elif k in ['q', 'S+q']: self.world.params['m'] += inc_10_or_1 * 0.001; self.check_auto_load()
		elif k in ['a', 'S+a']: self.world.params['m'] -= inc_10_or_1 * 0.001; self.check_auto_load()
		elif k in ['w', 'S+w']: self.world.params['s'] += inc_10_or_1 * 0.0001; self.check_auto_load()
		elif k in ['s', 'S+s']: self.world.params['s'] -= inc_10_or_1 * 0.0001; self.check_auto_load()
		elif k in ['t', 'S+t']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] // double_or_not - inc_or_not))
		elif k in ['g', 'S+g']: self.world.params['T'] = max(5, min(10000, self.world.params['T'] *  double_or_not + inc_or_not))
		elif k in ['r', 'S+r']: self.set_zoom(+inc_mul_or_not, +inc_or_not); self.transform_world()
		elif k in ['f', 'S+f']: self.set_zoom(-inc_mul_or_not, -inc_or_not); self.transform_world()
		elif k in ['C+y', 'S+C+y']: self.automaton.kn = (self.automaton.kn + inc_or_dec - 1) % len(self.automaton.kernel_core) + 1
		elif k in ['C+u', 'S+C+u']: self.automaton.gn = (self.automaton.gn + inc_or_dec - 1) % len(self.automaton.field_func) + 1
		elif k in ['C+i', 'S+C+i']: self.automaton.clip_mode = (self.automaton.clip_mode + inc_or_dec) % 2
		elif k in ['C+o']: self.automaton.is_multistep = not self.automaton.is_multistep
		elif k in ['C+p']: self.world.params['T'] *= -1; self.world.params['m'] = 1 - self.world.params['m']; self.world.cells = 1 - self.world.cells
		elif k in ['equal', 'S+plus']: self.colormap_id = (self.colormap_id + inc_or_dec) % len(self.colormap)
		elif k in ['tab', 'S+tab']: self.show_what = (self.show_what + inc_or_dec) % 5
		elif k in ['down',  'S+down']:  self.tx['shift'][0] += inc_10_or_1; self.transform_world()
		elif k in ['up',    'S+up']:    self.tx['shift'][0] -= inc_10_or_1; self.transform_world()
		elif k in ['right', 'S+right']: self.tx['shift'][1] += inc_10_or_1; self.transform_world()
		elif k in ['left',  'S+left']:  self.tx['shift'][1] -= inc_10_or_1; self.transform_world()
		elif k in ['prior', 'S+prior']: self.tx['rotate'] += inc_10_or_1; self.transform_world()
		elif k in ['next',  'S+next']:  self.tx['rotate'] -= inc_10_or_1; self.transform_world()
		elif k in ['home']: self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
		elif k in ['end']:  self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
		elif k in ['backspace', 'delete']: self.clear_world()
		elif k in ['z']: self.load_animal_id(self.animal_id)
		elif k in ['x']: self.load_part(self.fore, is_random=True, is_replace=False)
		elif k in ['c', 'S+c']: self.load_animal_id(self.animal_id - inc_1_or_10)
		elif k in ['v', 'S+v']: self.load_animal_id(self.animal_id + inc_1_or_10)
		elif k in [str(i) for i in range(10)] + ['S'+str(i) for i in range(10)]: self.load_animal_code(self.ANIMAL_KEY_LIST.get(k))
		elif k in ['m']: self.center_world()
		elif k in ['C+m']: self.center_world(); self.is_auto_center = not self.is_auto_center
		elif k in ['C+z']: self.is_auto_load = not self.is_auto_load
		elif k in ['C+c', 'S+C+c', 'C+s', 'S+C+s']:
			A = copy.deepcopy(self.world)
			A.crop(1/255)
			data = A.to_data(is_shorten='S+' not in k)
			if k.endswith('c'):
				self.clipboard_st = json.dumps(data, separators=(',', ':'))
				self.win.clipboard_clear()
				self.win.clipboard_append(self.clipboard_st)
				# print(self.clipboard_st)
				self.status = "> board saved to clipboard"
			elif k.endswith('s'):
				with open('last_animal.json', 'w') as file:
					json.dump(data, file, separators=(',', ':'))
				with open('last_animal.rle', 'w') as file:
					file.write('#N '+A.long_name()+'\n')
					file.write('x = '+str(A.cells.shape[1])+', y = '+str(A.cells.shape[0])+', rule = Lenia('+A.params2st()+')\n')
					file.write(data['cells'].replace('$','$\n')+'\n')
				self.status = "> board saved to files '{}' & '{}'".format('last_animal.json', 'last_animal.rle')
		elif k in ['C+v']:
			self.clipboard_st = self.win.clipboard_get()
			data = json.loads(self.clipboard_st)
			self.load_part(Board.from_data(data), zoom=1)
		elif k in ['C+x']: self.is_layered = not self.is_layered
		elif k in ['C+r', 'S+C+r']: self.recorder.toggle_recording(is_save_frames='S+' in k)
		elif k.endswith('_l') or k.endswith('_r'): is_ignore = True
		else: self.excess_key = k

		if not is_ignore and self.is_loop:
			self.world.params = {k:round(x, ROUND) if type(x)==float else x for (k,x) in self.world.params.items()}
			self.tx = {k:round(x, ROUND) if type(x)==float else x for (k,x) in self.tx.items()}
			self.automaton.calc_once(is_update=False)
			self.update_menu()
			self.update_info()

	# def toggle(self, k):
		# setattr(self, k, not getattr(self, k))
	# def rotate(self, k, d, len, base=0):
		# setattr(self, k, (getattr(self, k) + d - base) % len + base)
	def add_menu(self, text):
		m = tk.Menu(self.menu)
		self.menu_seq = 0
		self.menu.add_cascade(label=text, menu=m)
		return m
	def add_menu_item(self, m, text, acc, key):
		self.menu_seq += 1
		m.add_command(label=text, accelerator=acc, command=lambda:self.key_press(key) if key is not None else None)
	def add_menu_separator(self, m):
		self.menu_seq += 1
		m.add_separator()
	def add_menu_tick(self, m, text, var_name, acc, key):
		self.menu_seq += 1
		self.menu_vars[var_name] = tk.BooleanVar(value=getattr(self, var_name))
		m.add_checkbutton(label=text, accelerator=acc, command=lambda:self.key_press(key) if key is not None else None, variable=self.menu_vars[var_name])
	def add_param_display(self, m, param_name):
		self.menu_seq += 1
		self.menu_params.append(m._name + ':' + str(self.menu_seq) + ':' + param_name)
		m.add_command(background='navy', foreground='white')
	def add_value_display(self, m, var_name, acc=None, key=None):
		self.menu_seq += 1
		self.menu_values[var_name] = m._name + ':' + str(self.menu_seq) + ':' + var_name
		m.add_command(background='dark green', foreground='white', accelerator=acc, command=lambda:self.key_press(key) if key is not None else None)
	def display_menu_value(self, var_name, text, value):
		code = self.menu_values[var_name].split(':')
		self.menu.children[code[0]].entryconfig(int(code[1]), label='{0} [{1}]'.format(text, value))
	def update_menu(self):
		for var_name in self.menu_vars:
			self.menu_vars[var_name].set(getattr(self, var_name))
		for param in self.menu_params:
			code = param.split(':')
			self.menu.children[code[0]].entryconfig(int(code[1]), label='{0} = {1}'.format(code[2], '['+Board.fracs2st(self.world.params[code[2]])+']' if code[2] is 'b' else self.world.params[code[2]]))
		self.display_menu_value('kn', 'Kernal', ["Polynomial","Gaussian"][(self.world.params.get('kn') or self.automaton.kn) - 1])
		self.display_menu_value('gn', 'Field', ["Polynomial","Gaussian"][(self.world.params.get('gn') or self.automaton.gn) - 1])
		self.display_menu_value('clip', 'Clip', ["Hard","Soft"][self.automaton.clip_mode])
		self.display_menu_value('step', 'Step', ["Single","Multi"][self.automaton.is_multistep])
		self.display_menu_value('invert', 'Invert', ["Normal","Inverted"][self.world.params['T']<0])
		self.display_menu_value('color', 'Colormap', ["Blue/red","Green/purple","Red/green","Black/white"][self.colormap_id])
		self.display_menu_value('show', 'Display', ["World","Potential","Field","Change","Kernel"][self.show_what])
		self.display_menu_value('center', 'Auto center', ["Off","On"][self.is_auto_center])
		self.display_menu_value('load', 'Auto place/paste', ["Off","On"][self.is_auto_load])
		self.display_menu_value('layer', 'Layer mode', ["Off","On"][self.is_layered])
		self.display_menu_value('rec', 'Recorder', ["Off","On"][self.recorder.is_recording])
		self.display_menu_value('animal', '', self.world.long_name())

	def create_menu(self):
		self.menu_vars = {}
		self.menu_params = []
		self.menu_values = {}
		self.menu = tk.Menu(self.win)
		self.win.config(menu=self.menu)

		m = self.add_menu('Main')
		self.add_menu_tick(m, 'Start/Stop', 'is_run', 'Enter', 'enter')
		self.add_menu_item(m, 'Once', 'Space', 'space')
		self.add_menu_separator(m)
		self.add_value_display(m, 'load', 'ctrl+Z', 'C+z')
		self.add_value_display(m, 'layer', 'ctrl+X', 'C+x')
		self.add_menu_separator(m)
		self.add_menu_item(m, 'Copy', 'ctrl+C', 'C+c')
		self.add_menu_item(m, 'Paste', 'ctrl+V', 'C+v')
		self.add_menu_item(m, 'Save', 'ctrl+S', 'C+s')
		self.add_value_display(m, 'rec', 'ctrl+R', 'C+r')
		self.add_menu_separator(m)
		self.add_menu_item(m, 'Quit', 'Esc', 'escape')

		m = self.add_menu('World')
		self.add_menu_item(m, 'Clear', 'Bkspc', 'backspace')
		self.add_menu_separator(m)
		self.add_menu_item(m, 'Center', 'M', 'm')
		self.add_value_display(m, 'center', 'ctrl+M', 'C+m')
		self.add_menu_separator(m)
		self.add_menu_item(m, '(Nudge)', '(shift+)', None)
		self.add_menu_item(m, 'Move up', 'Up', 'up')
		self.add_menu_item(m, 'Move down', 'Down', 'down')
		self.add_menu_item(m, 'Move left', 'Left', 'left')
		self.add_menu_item(m, 'Move right', 'Right', 'right')
		self.add_menu_item(m, 'Rotate clockwise', 'PgDn', 'next')
		self.add_menu_item(m, 'Rotate anti-clockwise', 'PgUp', 'prior')
		self.add_menu_item(m, 'Flip vertically', 'Home', 'home')
		self.add_menu_item(m, 'Flip horizontally', 'End', 'end')
		self.add_menu_separator(m)
		self.add_value_display(m, 'color', '=', 'equal')
		self.add_value_display(m, 'show', 'Tab', 'tab')

		m = self.add_menu('Animal')
		self.add_value_display(m, 'animal')
		self.add_menu_item(m, 'Place at center', 'Z', 'z')
		self.add_menu_item(m, 'Place at random', 'X', 'x')
		self.add_menu_item(m, 'Previous animal', 'C', 'c')
		self.add_menu_item(m, 'Next animal', 'V', 'v')
		self.add_menu_item(m, 'Previous 10', 'shift+C', 'S+c')
		self.add_menu_item(m, 'Next 10', 'shift+V', 'S+v')
		self.add_menu_separator(m)
		for (key,code) in self.ANIMAL_KEY_LIST.items():
			id = self.get_animal_id(code)
			if id is not None:
				self.add_menu_item(m, self.animal_data[id]['name'], key.replace('S+','shift+'), key)

		m = self.add_menu('Params')
		self.add_param_display(m, 'm')
		self.add_menu_item(m, 'm + 0.01', 'Q', 'q')
		self.add_menu_item(m, 'm - 0.01', 'A', 'a')
		self.add_menu_item(m, 'm + 0.001', 'shift+Q', 'S+q')
		self.add_menu_item(m, 'm - 0.001', 'shift+A', 'S+a')
		self.add_param_display(m, 's')
		self.add_menu_item(m, 's + 0.001', 'W', 'w')
		self.add_menu_item(m, 's - 0.001', 'S', 's')
		self.add_menu_item(m, 's + 0.0001', 'shift+W', 'S+w')
		self.add_menu_item(m, 's - 0.0001', 'shift+S', 'S+s')
		self.add_param_display(m, 'R')
		self.add_menu_item(m, 'Zoom in', 'R', 'r')
		self.add_menu_item(m, 'Zoom out', 'F', 'f')
		self.add_menu_item(m, 'R + 1', 'shift+R', 'S+r')
		self.add_menu_item(m, 'R - 1', 'shift+F', 'S+f')
		self.add_param_display(m, 'T')
		self.add_menu_item(m, 'Double speed', 'T', 't')
		self.add_menu_item(m, 'Half speed', 'G', 'g')
		self.add_menu_item(m, 'T + 1', 'shift+T', 'S+t')
		self.add_menu_item(m, 'T - 1', 'shift+G', 'S+g')
		self.add_param_display(m, 'b')
		self.add_menu_separator(m)
		self.add_value_display(m, 'kn', 'ctrl+Y', 'C+y')
		self.add_value_display(m, 'gn', 'ctrl+U', 'C+u')
		self.add_value_display(m, 'clip', 'ctrl+I', 'C+i')
		self.add_value_display(m, 'step', 'ctrl+O', 'C+o')
		self.add_value_display(m, 'invert', 'ctrl+P', 'C+p')

	def update_info(self):
		print(self.status)

if __name__ == '__main__':
	lenia = Lenia()
	lenia.load_animal_code(lenia.ANIMAL_KEY_LIST['1'], zoom=8)
#	lenia.load_animal_code(lenia.ANIMAL_KEY_LIST['2'], zoom=1)
	lenia.update_menu()
	lenia.loop()

# GPU FFT
# https://pythonhosted.org/pyfft/
# http://arrayfire.org/docs/group__signal__func__fft2.htm
