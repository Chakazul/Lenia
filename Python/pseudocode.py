

	self.world = Board(size=(SIZEX, SIZEY))

	self.calc_once()
	self.world.angle = 0
	self.world.shift = [0,0]
	self.world.new_R = self.world.params['R']

	if is_run: self.world.take_snapshot()
	self.world.angle += 10
	self.world.shift += [0,10]
	self.world.new_R *= 2
	self.world.transform()


	self.animal = Board(data=self.animals[self.animal_ID])
	self.animal.take_snapshot()

	self.animal.angle += 10
	self.animal.new_R *= 2
	self.animal.transform()
	self.animal.shift += [0,10]
	self.world.add(self.animal, offset=self.animal.shift)
