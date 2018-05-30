import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

global SIZE, MID
SIZE = 1 << 8
MID = int(SIZE / 2)
BORDER = 0.03

class Lenia:
	def __init__(self):
		mpl.rcParams['toolbar'] = 'None'
		self.is_run = True
		self.is_once = False
		self.kernel_core = {
			0: lambda r: (4 * r * (1-r))**4,
			1: lambda r: np.exp( 4 - 1 / (r * (1-r)) )
		}
		self.delta_func = {
			0: lambda n: np.maximum(0, 1 - (n - self.params['m'])**2 / (self.params['s']**2 * 9) )**4 * 2 - 1,
			1: lambda n: np.exp( - (n - self.params['m'])**2 / (self.params['s']**2 * 2) ) * 2 - 1
		}
		self.params = {'R':11, 'b':[1], 'm':0.15, 's':0.015, 'dt':0.1, 'kn':0, 'dn':0}
		self.calc_kernel()
		self.clear_world()
		self.create_figure(self.world)
		self.create_panel(0, 0, 0, self.world, 0, 1)
		self.create_panel(1, 0, 1, self.potential, 0, 0.3)
		self.create_panel(2, 1, 0, self.delta, -1, 1)
		self.create_panel(3, 1, 1, self.kernel, 0, 1)

	def key_press(self, event):
		k = event.key.lower()
		if k=='escape': plt.close()
		elif k=='enter': self.is_run = not self.is_run
		elif k==' ': self.is_once = not self.is_once
		elif k=='t': self.params['dt'] = self.params['dt'] * 2
		elif k=='g': self.params['dt'] = self.params['dt'] / 2
		elif k=='1': self.load_cells(1, multiply=4)
		elif k=='2': self.load_cells(2, multiply=4)
		elif k in ['ctrl+c', 'cmd+c']:
			cells_list = np.rint(self.get_cropped(self.world)*100).astype(int).tolist()
			data = {'name':self.name, 'params':self.params, 'cells':cells_list}
			self.clip_st = json.dumps(data, separators=(',', ':'))
			print(self.clip_st)
			with open('last_animal.json', 'w') as file:
				json.dump(data, file, separators=(',', ':'))
		elif k in ['ctrl+v', 'cmd+v']:
			data = json.loads(self.clip_st)
			self.name = data['name']
			self.params = data['params']
			self.cells = np.array(data['cells'])/100
			self.clear_world()
			self.calc_kernel()
			self.add_cells(self.cells)
		else: print(k)

	def create_figure(self, A):
		self.dpi = 80
		self.img_size = np.shape(A)[0]/self.dpi
		self.fig_size = self.img_size * 2 + BORDER * 3
		self.fig = plt.figure(figsize=(self.fig_size, self.fig_size), dpi=self.dpi)
		self.fig.canvas.set_window_title("Lenia")
		self.fig.canvas.mpl_connect('key_press_event', self.key_press)
		self.ax = [None] * 4
		self.img = [None] * 4

	def create_panel(self, i, c, r, A, vmin=0, vmax=1):
		self.ax[i] = self.fig.add_axes([
			((c*self.img_size+(c+1)*BORDER)) / self.fig_size, 
			((r*self.img_size+(r+1)*BORDER)) / self.fig_size,
			self.img_size / self.fig_size, self.img_size / self.fig_size])
		self.ax[i].axis('off')
		self.img[i] = self.ax[i].imshow(A, cmap='jet', interpolation='none', aspect=1, vmin=vmin, vmax=vmax)
		self.fig.canvas.draw()

	def show_world(self):
		self.img[0].set_array(self.world)
		self.img[1].set_array(self.potential)
		self.img[2].set_array(self.delta)
		if not self.kernel_updated:
			self.img[3].set_array(self.kernel)
			self.kernel_updated = True

	def kernel_shell(self, r):
		k = len(self.params['b'])
		kr = k * r
		bs = np.array(self.params['b'])
		b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
		kfunc = self.kernel_core[self.params['kn']]
		return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b

	def calc_once(self):
		world_FFT = np.fft.fft2(self.world)
		self.potential = np.roll(np.real(np.fft.ifft2(self.kernel_FFT * world_FFT)), MID, (0, 1))
		dfunc = self.delta_func[self.params['dn']]
		self.delta = dfunc(self.potential)
		self.world = np.clip(self.world + self.delta * self.params['dt'], 0, 1)

	def calc_kernel(self):
		I = np.array([np.arange(SIZE),]*SIZE)
		X = (I-MID) / self.params['R']
		Y = X.T
		D = np.sqrt(X**2 + Y**2)

		self.kernel = self.kernel_shell(D)
		self.kernel_sum = np.sum(self.kernel)
		kernel_norm = self.kernel / self.kernel_sum
		self.kernel_FFT = np.fft.fft2(kernel_norm)
		self.kernel_updated = False

	def load_cells(self, id, is_clear=True, multiply=1):
		if is_clear:
			self.clear_world()
		if id==1: self.name = 'Orbium bicaudatus'; self.params = {'R':13,'b':[1],'m':0.15,'s':0.014,'dt':0.1,'kn':0,'dn':0}; self.cells = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],[0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],[0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],[0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],[0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],[0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],[0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],[0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],[0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],[0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],[0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],[0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],[0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],[0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],[0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],[0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],[0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],[0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],[0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],[0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
		elif id==2: self.name = 'Rotorbium'; self.params = {'R':13,'b':[1],'m':0.156,'s':0.0224,'dt':0.1,'kn':0,'dn':0}; self.cells = np.array([[0,0,0,0,0,0,0,0,0.003978,0.016492,0.004714,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.045386,0.351517,0.417829,0.367137,0.37766,0.426948,0.431058,0.282864,0.081247,0,0,0,0,0,0],[0,0,0,0,0.325473,0.450995,0.121737,0,0,0,0.003113,0.224278,0.47101,0.456459,0.247231,0.071609,0.013126,0,0,0],[0,0,0,0.386337,0.454077,0,0,0,0,0,0,0,0.27848,0.524466,0.464281,0.242651,0.096721,0.038476,0,0],[0,0,0.258817,0.583802,0.150994,0,0,0,0,0,0,0,0.226639,0.548329,0.550422,0.334764,0.153108,0.087049,0.042872,0],[0,0.008021,0.502406,0.524042,0.059531,0,0,0,0,0,0,0.033946,0.378866,0.615467,0.577527,0.357306,0.152872,0.090425,0.058275,0.023345],[0,0.179756,0.596317,0.533619,0.162612,0,0,0,0,0.015021,0.107673,0.325125,0.594765,0.682434,0.594688,0.381172,0.152078,0.073544,0.054424,0.030592],[0,0.266078,0.614339,0.605474,0.379255,0.195176,0.16516,0.179148,0.204498,0.299535,0.760743,1,1,1,1,0.490799,0.237826,0.069989,0.043549,0.022165],[0,0.333031,0.64057,0.686886,0.60698,0.509866,0.450525,0.389552,0.434978,0.859115,0.94097,1,1,1,1,1,0.747866,0.118317,0.037712,0.006271],[0,0.417887,0.6856,0.805342,0.824229,0.771553,0.69251,0.614328,0.651704,0.843665,0.910114,1,1,0.81765,0.703404,0.858469,1,0.613961,0.035691,0],[0.04674,0.526827,0.787644,0.895984,0.734214,0.661746,0.670024,0.646184,0.69904,0.723163,0.682438,0.618645,0.589858,0.374017,0.30658,0.404027,0.746403,0.852551,0.031459,0],[0.130727,0.658494,0.899652,0.508352,0.065875,0.009245,0.232702,0.419661,0.461988,0.470213,0.390198,0.007773,0,0.010182,0.080666,0.17231,0.44588,0.819878,0.034815,0],[0.198532,0.810417,0.63725,0.031385,0,0,0,0,0.315842,0.319248,0.321024,0,0,0,0,0.021482,0.27315,0.747039,0,0],[0.217619,0.968727,0.104843,0,0,0,0,0,0.152033,0.158413,0.114036,0,0,0,0,0,0.224751,0.647423,0,0],[0.138866,1,0.093672,0,0,0,0,0,0.000052,0.015966,0,0,0,0,0,0,0.281471,0.455713,0,0],[0,1,0.145606,0.005319,0,0,0,0,0,0,0,0,0,0,0,0.016878,0.381439,0.173336,0,0],[0,0.97421,0.262735,0.096478,0,0,0,0,0,0,0,0,0,0,0.013827,0.217967,0.287352,0,0,0],[0,0.593133,0.2981,0.251901,0.167326,0.088798,0.041468,0.013086,0.002207,0.009404,0.032743,0.061718,0.102995,0.1595,0.24721,0.233961,0.002389,0,0,0],[0,0,0.610166,0.15545,0.200204,0.228209,0.241863,0.243451,0.270572,0.446258,0.376504,0.174319,0.154149,0.12061,0.074709,0,0,0,0,0],[0,0,0.354313,0.32245,0,0,0,0.151173,0.479517,0.650744,0.392183,0,0,0,0,0,0,0,0,0],[0,0,0,0.329339,0.328926,0.176186,0.198788,0.335721,0.534118,0.549606,0.361315,0,0,0,0,0,0,0,0,0],[0,0,0,0,0.090407,0.217992,0.190592,0.174636,0.222482,0.375871,0.265924,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0.050256,0.235176,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.180145,0.132616,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0.092581,0.188519,0.118256,0,0,0,0]]) 
		if multiply > 1:
			self.double_cells(multiply)
		self.calc_kernel()
		self.add_cells(self.cells)

	def clear_world(self):
		self.world = np.zeros((SIZE, SIZE))
		self.potential = np.zeros((SIZE, SIZE))
		self.delta = np.zeros((SIZE, SIZE))

	def add_cells(self, cells):
		w = np.shape(cells)[1]
		h = np.shape(cells)[0]
		i = MID - int(w / 2)
		j = MID - int(h / 2)
		self.world[j:j+h, i:i+w] = cells

	def double_cells(self, n=2):
		w = np.shape(self.cells)[1] * n
		h = np.shape(self.cells)[0] * n
		cells2 = np.zeros((h, w))
		for i in range(n):
			for j in range(n):
				cells2[i:h:n, j:w:n] = self.cells
		self.cells = cells2
		self.params['R'] = self.params['R'] * n

	def get_cropped(self, A):
		coords = np.argwhere(A > 0)
		y0, x0 = coords.min(axis=0)
		y1, x1 = coords.max(axis=0) + 1
		return A[y0:y1, x0:x1]

	def run(self):
		while True:
			if self.is_run or self.is_once:
				self.calc_once()
				self.show_world()
				self.is_once = False
			plt.pause(0.000000000001)

lenia = Lenia()
lenia.load_cells(1, multiply=4)
lenia.run()

# GPU FFT
# https://pythonhosted.org/pyfft/
# http://arrayfire.org/docs/group__signal__func__fft2.htm
