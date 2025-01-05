class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
		"""Linear interpolation between initial_p and final_p over"""
		self.schedule_timesteps = schedule_timesteps
		self.initial_p = initial_p
		self.final_p = final_p

	def value(self, t):
		fraction = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)