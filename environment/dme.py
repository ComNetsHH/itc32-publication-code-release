from environment.channel import *
import util.verbose_print

class DMEGroundStation(InteractiveChannelModel.User):
	def __init__(self, interrogator_channel_index, response_channel_index, x, y, channel):
		InteractiveChannelModel.User.__init__(self, x, y, channel)
		self.interrogator_channel_index = interrogator_channel_index
		self.response_channel_index = response_channel_index
		self.aircraft = None
		self.next_response_due = None
		self.response_delay = 0.05  # 50 microseconds after the reception of an interrogation signal, the response is sent

	def request(self, aircraft):
		"""
		A DME aircraft transmits a request.
		:param aircraft:
		"""
		# Calculate distance between aircraft and ground station.
		distance = self.channel.__euclidean_distance__(self, aircraft)
		# Calculate propagation delay.
		propagation_delay = self.channel.__get_propagation_delay__(distance)
		# Set counter (millisecond-resolution) until request signal arrives and response delay has passed.
		self.next_response_due = propagation_delay + self.response_delay
		util.verbose_print.vprint("DME request sent by aircraft, arrives in " + str(propagation_delay) + "ms @groundstation (" + str(distance/1000) + "km).")
		# Put the signal on the radio channel.
		self.channel.access(self.interrogator_channel_index, aircraft)

	def update(self):
		if self.next_response_due is not None:
			self.next_response_due = self.next_response_due - 1
			if self.next_response_due <= 1.0:  # <=1 means within this timeslot
				self.handle_request()

	def handle_request(self):
		"""
		A request signal arrives at the ground station.
		"""
		util.verbose_print.vprint("DME response sent")
		# Send the response.
		# offset is the remaining time until the request actually arrives,
		# leaving it out causes rounding errors that lead to a response arriving at the user BEFORE the request
		# which is physically impossible...
		# e.g. DME_AS -> GS is t_prop=1.9, and GS -> USER is also t_prop=1.9
		# so DME_AS -> USER is t_prop=1.9+1.9=3.8, rounded down means it arrives in the 3rd timeslot from now
		# but if the .9 offset is left out, we have 1+1.9=2.9, rounded down is 2 timeslots away
		self.channel.access(self.response_channel_index, self, offset=self.next_response_due)
		self.next_response_due = None


class Aircraft(InteractiveChannelModel.User):
	def __init__(self, x, y, channel):
		InteractiveChannelModel.User.__init__(self, x , y, channel)
		self.speed = 0
		self.direction = 1
		self.speed_per_timeslot = 0
		self.current_timeslot = 0

	def set_speed(self, speed, direction):
		"""
		:param speed: In km/h.
		:param direction: 1 in x-direction, -1 in -x-direction.
		"""
		self.speed = speed
		self.direction = direction
		self.speed_per_timeslot = speed * 1000 / 60 / 60 / 1000  # meters per millisecond

	def update(self):
		self.current_timeslot = self.current_timeslot + 1
		self.x = self.x + self.direction*self.speed_per_timeslot


class DMEAircraft(Aircraft):
	def __init__(self, x, y, channel, request_frequency, ground_station):
		"""
		:param request_frequency: DME request frequency in milliseconds.
		"""
		Aircraft.__init__(self, x, y, channel)
		self.timeslots_until_next_request = request_frequency
		self.request_frequency = request_frequency
		self.ground_station = ground_station

	def update(self):
		Aircraft.update(self)
		self.timeslots_until_next_request = self.timeslots_until_next_request - 1
		if self.timeslots_until_next_request <= 0:
			self.timeslots_until_next_request = self.request_frequency
			self.transmit_dme_request()

	def transmit_dme_request(self):
		self.ground_station.request(self)
