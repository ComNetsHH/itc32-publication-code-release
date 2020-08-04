import unittest
from environment.dme import *
import util.verbose_print
from predictor.neural_network import *

class TestDME(unittest.TestCase):
	def setUp(self):
		self.num_channels = 2
		self.interrogator_channel_index = 0
		self.response_channel_index = 1
		self.user = Aircraft(0, 0, None)
		self.max_timeslots = 1000
		self.channel = InteractiveChannelModel(self.num_channels, self.user, self.max_timeslots)
		self.user.set_channel(self.channel)
		self.ground_station = DMEGroundStation(self.interrogator_channel_index, self.response_channel_index, 260*1000, 150*1000, self.channel)
		self.dme_request_frequency = 5 # send a DME request every x ms
		self.dme_aircraft = DMEAircraft(260*1000, 300*1000, self.channel, self.dme_request_frequency, self.ground_station)
		self.user.set_speed(800, 1)
		self.dme_aircraft.set_speed(800, -1)
		util.verbose_print.verbose = True

	def test_request_response(self):
		num_timeslots = self.dme_request_frequency + 5
		counter = 0
		tprop_as_user = self.channel.__get_propagation_delay__(self.channel.__euclidean_distance__(self.dme_aircraft, self.user))
		tprop_as_gs = self.channel.__get_propagation_delay__(self.channel.__euclidean_distance__(self.dme_aircraft, self.ground_station))
		tprop_gs_user = self.channel.__get_propagation_delay__(self.channel.__euclidean_distance__(self.ground_station, self.user))
		expected_signal_arrival_request = self.dme_request_frequency - 1 + math.floor(tprop_as_user)  # -1 due to zero-based indexing (sending request in 10th slot means index 9)
		expected_signal_arrival_response = self.dme_request_frequency - 1 + math.floor(tprop_as_gs + tprop_gs_user)
		print(expected_signal_arrival_request)
		print(expected_signal_arrival_response)
		print(tprop_as_user)
		print(tprop_as_gs)
		print(tprop_gs_user)

		for timeslot in range(num_timeslots):
			util.verbose_print.vprint("t=" + str(timeslot+1) + " / " + str(num_timeslots))
			counter = counter + 1
			self.ground_station.update()
			self.dme_aircraft.update()
			self.user.update()
			self.channel.update()

		channel_state_matrix = self.channel.state_matrix
		for timeslot in range(num_timeslots):
			print(str(timeslot) + " " + str(channel_state_matrix[timeslot]))
		self.assertEqual(channel_state_matrix[expected_signal_arrival_request, self.interrogator_channel_index], 1)
		self.assertEqual(channel_state_matrix[expected_signal_arrival_response, self.response_channel_index], 1)

	def test_mobility(self):
		self.dme_aircraft.set_speed(20000, -1)
		num_timeslots = 1000
		last_distance = 0
		for timeslot in range(num_timeslots):
			self.dme_aircraft.update()
			self.ground_station.update()
			self.user.update()
			distance = self.channel.__euclidean_distance__(self.user, self.dme_aircraft)
			if timeslot > 0:
				self.assertLess(distance, last_distance)
			last_distance = distance

	def test_max_distance_exceeded(self):
		self.dme_aircraft.x = 520*1000

		# GS >300km from user
		self.assertGreater(self.channel.__euclidean_distance__(self.ground_station, self.user), 300.0*1000)
		self.assertGreater(self.channel.__euclidean_distance__(self.ground_station, self.dme_aircraft), 300.0*1000)
		self.assertGreater(self.channel.__euclidean_distance__(self.dme_aircraft, self.user), 600.0*1000)
		self.assertGreaterEqual(self.channel.__get_propagation_delay__(self.channel.__euclidean_distance__(self.ground_station, self.user)), 1.0)
		self.assertGreaterEqual(self.channel.__get_propagation_delay__(self.channel.__euclidean_distance__(self.dme_aircraft, self.user)), 2.0)

		num_timeslots = 15
		for timeslot in range(num_timeslots):
			self.dme_aircraft.update()
			self.channel.update()
			self.ground_station.update()
			self.user.update()
			state_vector = self.channel.get_state_vector()
			num_signals_arrive = np.sum(state_vector)
			vprint("t=" + str(self.channel.current_timeslot) + " #signals=" + str(num_signals_arrive))

		state_matrix = self.channel.state_matrix
		self.assertEqual(state_matrix[self.dme_request_frequency+1, 1], 1)  # tx in slot 5, arrival in slot 6
		self.assertEqual(state_matrix[self.dme_request_frequency+1+self.dme_request_frequency, 1], 1)
		self.assertEqual(state_matrix[self.dme_request_frequency+1+2*self.dme_request_frequency, 1], 1)
		self.assertEqual(np.sum(state_matrix[:,1]), 3)
		self.assertEqual(np.sum(state_matrix[:,0]), 0)  # no interrogation signals arrive due to excessive distance


	def test_two_signals(self):
		self.dme_request_frequency = 5
		self.ground_station = DMEGroundStation(self.interrogator_channel_index, self.response_channel_index, 260*1000, 150*1000, self.channel)
		self.user = Aircraft(260*1000, 0, self.channel)
		self.user.set_speed(800, 1)
		self.dme_aircraft = DMEAircraft(260*1000, 300*1000, self.channel, self.dme_request_frequency, self.ground_station)
		self.dme_aircraft.set_speed(800, -1)

		# DME user is 150km "above" ground station, user 150km "below" (according to y-coord)
		self.assertGreaterEqual(self.channel.__euclidean_distance__(self.ground_station, self.user), 150.0*1000)
		self.assertGreaterEqual(self.channel.__euclidean_distance__(self.ground_station, self.dme_aircraft), 150.0*1000)
		self.assertGreaterEqual(self.channel.__euclidean_distance__(self.dme_aircraft, self.user), 300.0*1000)

		num_timeslots = 15
		for timeslot in range(num_timeslots):
			self.dme_aircraft.update()
			self.channel.update()
			self.ground_station.update()
			self.user.update()
			state_vector = self.channel.get_state_vector()
			num_signals_arrive = np.sum(state_vector)
			vprint("t=" + str(self.channel.current_timeslot) + " #signals=" + str(num_signals_arrive))

		state_matrix = self.channel.state_matrix
		self.assertEqual(state_matrix[self.dme_request_frequency, 1], 1)  # tx in slot 5, arrival in slot 5
		self.assertEqual(state_matrix[2*self.dme_request_frequency, 1], 1)
		self.assertEqual(state_matrix[3*self.dme_request_frequency, 1], 1)
		self.assertEqual(np.sum(state_matrix[:,1]), 3)
		self.assertEqual(state_matrix[self.dme_request_frequency, 0], 1)  # tx in slot 5, arrival in slot 5
		self.assertEqual(state_matrix[2*self.dme_request_frequency, 0], 1)
		self.assertEqual(state_matrix[3*self.dme_request_frequency, 0], 1)
		self.assertEqual(np.sum(state_matrix[:,0]), 3)  # all interrogation signals arrive at the user, too


	def test_state_vec_as_input_into_lstm(self):
		num_timeslots = self.dme_request_frequency * 3
		last_state_vec = None

		sample_length = 1
		learning_rate = 0.005
		num_hidden_layers = 2
		num_neurons = [200, 150]
		neural_network = LSTMNetwork(self.num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)

		for timeslot in range(num_timeslots):
			self.dme_aircraft.update()
			self.channel.update()
			self.ground_station.update()
			self.user.update()

			current_state_vec = self.channel.get_state_vector()
			if timeslot > 0:
				label = np.reshape(current_state_vec, (1, 1, self.num_channels))
				input = np.reshape(last_state_vec, (1, 1, self.num_channels))
				vprint("t=" + str(timeslot))
				vprint(input)
				vprint(label)
				vprint("")
				neural_network.get_keras_model().fit(x=input, y=label, verbose=False)

			last_state_vec = current_state_vec