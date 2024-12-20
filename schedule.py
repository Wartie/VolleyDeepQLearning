class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """Value of the schedule at time t"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
        
class RewardBasedEGreedy(object):
	def __init__(self, total_time, final_p, initial_p=1.0):
		self.E = total_time
		self.e_min = final_p
		self.e_init = initial_p
		self.e_current = self.e_init
		self.R = -1.0
		self.I = 0.001
		
	def value(self, G, last_reward):
		for e in range(1, self.E):
			if self.e_current > self.e_min and last_reward >= self.R:
				self.e_current = (self.e_init - self.e_min) * max((self.E-G)/self.E, 0)
				self.R += self.I
		return self.e_current
