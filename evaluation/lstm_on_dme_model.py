import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from environment.channel import *
from environment.dme import *
import matplotlib.pyplot as plt
import util.verbose_print
from predictor.neural_network import *
from util.callback_loss_history import *
from util.confidence_intervals import *
import tensorflow.compat.v1 as tf

def shift(arr, shift, fill_value=np.nan):
	"""
	Shifts array 'arr' contents by 'shift'.
	:param arr:
	:param shift:
	:param fill_value:
	:return:
	"""
	result = np.empty_like(arr)
	if shift > 0:
		result[:shift] = fill_value
		result[shift:] = arr[:-shift]
	elif shift < 0:
		result[shift:] = fill_value
		result[:shift] = arr[-shift:]
	else:
		result[:] = arr
	return result


def get_training_data(x_as, y_as, speed1, x_gs, y_gs, x_as2, y_as2, speed2, interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length):
	"""
	:param x_as: Secondary user aircraft x-position.
	:param y_as: Secondary user aircraft y-position.
	:param speed1: Secondary user aircraft speed.
	:param x_gs: Ground station x-position.
	:param y_gs: Ground station y-position.
	:param x_as2: Primary user aircraft x-position.
	:param y_as2: Primary user aircraft y-position.
	:param speed2: Primary user aircraft speed.
	:param interrogator_channel_index: Index of frequency channel DME requests are transmitted on.
	:param response_channel_index: Index of frequency channel DME responses are transmitted on.
	:param dme_request_frequency: Number of milliseconds until next DME request.
	:param training_time: Number of timeslots observations should be made to make up the training data.
	:param validation_time: Number of timeslots observations should be made to make up the validation data.
	:param num_channels: Number of logical frequency channels.
	:param sample_length: Number of timeslots to aggregate into one sample.
	:return:
	"""
	simtime_max = training_time + validation_time
	user = Aircraft(x_as, y_as, None)
	user.set_speed(speed1[0], speed1[1])
	channel = InteractiveChannelModel(num_channels, user, simtime_max)
	user.set_channel(channel)
	ground_station = DMEGroundStation(interrogator_channel_index, response_channel_index, x_gs, y_gs, channel)
	dme_aircraft = DMEAircraft(x_as2, y_as2, channel, dme_request_frequency, ground_station)
	dme_aircraft.set_speed(speed2[0], speed2[1])
	num_training_samples = int(training_time - sample_length)
	num_validation_samples = int(validation_time - sample_length)

	for timeslot in range(simtime_max):
		dme_aircraft.update()
		channel.update()
		ground_station.update()
		user.update()
	training_timeslots = channel.state_matrix[0:training_time]
	training_data = np.zeros((num_training_samples, sample_length, num_channels))
	training_labels = np.zeros((num_training_samples, num_channels))
	current_sample = training_timeslots[0:sample_length]
	training_data[0] = current_sample
	training_labels[0] = training_timeslots[sample_length]
	for i in range(sample_length, training_time - 1):
		for j in range(1, sample_length):
			current_sample[j-1] = current_sample[j]
		current_sample[-1] = training_timeslots[i]
		training_data[i-sample_length+1] = current_sample
		training_labels[i-sample_length+1] = training_timeslots[i+1]

	validation_timeslots = channel.state_matrix[training_time:simtime_max]
	validation_data = np.zeros((num_validation_samples, sample_length, num_channels))
	validation_labels = np.zeros((num_validation_samples, num_channels))
	current_sample = validation_timeslots[0:sample_length]
	validation_data[0] = current_sample
	validation_labels[0] = validation_timeslots[sample_length]
	for i in range(sample_length, validation_time - 1):
		for j in range(1, sample_length):
			current_sample[j-1] = current_sample[j]
		current_sample[-1] = validation_timeslots[i]
		if sample_length > 1:
			validation_data[i-sample_length+1] = current_sample
			validation_labels[i-sample_length+1] = validation_timeslots[i+1]
		else:
			validation_data[i-sample_length] = current_sample
			validation_labels[i-sample_length] = validation_timeslots[i+1]

	return training_data, training_labels, validation_data, validation_labels


def validate_training_data():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5  # send a DME request every x ms
	training_time = 11
	validation_time = 11
	sample_length = 2  # milliseconds

	x_gs = 260*1000
	y_gs = 150*1000
	# 0.5ms propagation delay to ground station
	x_as_1 = 260*1000
	y_as_1 = 0
	# 0.5ms propagation delay to ground station
	x_dme_1 = 260*1000
	y_dme_1 = 300*1000

	training_data, training_labels, validation_data, validation_labels = get_training_data(x_as_1, y_as_1, [800, 1], x_gs, y_gs, x_dme_1, y_dme_1, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length)
	# print(training_data)
	# print(training_labels)
	# print("\n")
	# print(validation_data)
	# print(validation_labels)
	assert(training_data[0:4].any() == 0)
	assert(training_data[6:9].any() == 0)
	assert(training_data[4][0].any() == 0)
	assert(training_data[4][1].any() == 1)
	assert(training_data[5][0].any() == 1)
	assert(training_data[5][1].any() == 0)

	assert(validation_data[0:3].any() == 0)
	assert(validation_data[5:8].any() == 0)
	assert(validation_data[3][0].any() == 0)
	assert(validation_data[3][1].any() == 1)
	assert(validation_data[4][0].any() == 1)
	assert(validation_data[4][1].any() == 0)
	assert(validation_data[8][0].any() == 0)
	assert(validation_data[8][1].any() == 1)

	assert(training_labels[0:3].any() == 0)
	assert(training_labels[5:8].any() == 0)
	assert(training_labels[3].any() == 1)
	assert(training_labels[8].any() == 1)

	assert(validation_labels[0:2].any() == 0)
	assert(validation_labels[2].any() == 1)
	assert(validation_labels[3:7].any() == 0)
	assert(validation_labels[7].any() == 1)
	assert(validation_labels[8].any() == 0)


def validate_training_data_sample_length_1():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5  # send a DME request every x ms
	training_time = 11
	validation_time = 11
	sample_length = 1  # milliseconds

	x_gs = 260*1000
	y_gs = 150*1000
	# 0.5ms propagation delay to ground station
	x_as_1 = 260*1000
	y_as_1 = 0
	# 0.5ms propagation delay to ground station
	x_dme_1 = 260*1000
	y_dme_1 = 300*1000

	training_data, training_labels, validation_data, validation_labels = get_training_data(x_as_1, y_as_1, [800, 1], x_gs, y_gs, x_dme_1, y_dme_1, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length)
	print(training_data)
	print(training_labels)
	print("\n")
	print(validation_data)
	print(validation_labels)


def plot_learning_two_signals():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5 # send a DME request every x ms
	simtime_max = 5000
	sample_length = dme_request_frequency  # milliseconds
	validation_time = int(simtime_max/10)
	while validation_time % sample_length != 0:
		validation_time = validation_time + 1
		simtime_max = simtime_max + 1
	util.verbose_print.verbose = False
	num_training_samples = int(simtime_max / sample_length)

	training_data, training_labels, _, _ = get_training_data(260*1000, 0, [800, 1], 260*1000, 150*1000, 260*1000, 300*1000, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, simtime_max, 0, num_channels, sample_length)

	plt.rcParams.update({'font.size': 28})
	plt.ylabel('channel index')
	plt.xlabel('time slot')
	observations = np.zeros((sample_length*5, num_channels))
	for sample in range(5):
		for i in range(sample_length):
			observations[sample*sample_length + i] = training_data[sample,i,:]
	plt.imshow(np.transpose(observations), cmap='Greys')
	filename = "_imgs/dme/lstm/channel_access_pattern.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 4), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)

	learning_rate = 0.0005
	num_hidden_layers = 2
	num_neurons = [200, 150]

	num_repetitions = 20
	accuracy_mat = np.zeros((num_repetitions, num_training_samples))
	for rep in range(num_repetitions):
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=False)
		accuracy_vec = BinaryAccuracyHistory()
		neural_network.get_keras_model().fit(training_data, training_labels, batch_size=1, callbacks=[accuracy_vec])
		accuracy_mat[rep] = accuracy_vec.accuracies

	# Compute batch-means for every data point.
	batch_means_split = 4
	batch_means = columnwise_batch_means(accuracy_mat, batch_means_split)
	# Compute range for each data point using confidence intervals.
	sample_means = np.zeros(num_training_samples)
	sample_means_minus = np.zeros(num_training_samples)
	sample_means_plus = np.zeros(num_training_samples)
	confidence = 0.95
	for data_point in range(num_training_samples):
		sample_means[data_point], sample_means_minus[data_point], sample_means_plus[data_point] = calculate_confidence_interval(batch_means[:,data_point], confidence)

	x = range(len(sample_means))
	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Sample [#]')
	plt.ylabel('Accuracy')
	plt.plot(x, sample_means)
	plt.fill_between(x, sample_means_minus, sample_means_plus, alpha=0.5)
	filename = "_imgs/dme/lstm/lstm_training_accuracy.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 10), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


def verify_prediction():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5 # send a DME request every x ms
	simtime_max = 1000
	sample_length = dme_request_frequency  # milliseconds
	validation_time = int(simtime_max/10)
	while validation_time % sample_length != 0:
		validation_time = validation_time + 1
		simtime_max = simtime_max + 1

	util.verbose_print.verbose = False

	training_data, training_labels, validation_data, validation_labels = get_training_data(260*1000, 0, [800, 1], 260*1000, 150*1000, 260*1000, 300*1000, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, simtime_max-validation_time, validation_time, num_channels, sample_length)

	learning_rate = 0.0005
	num_hidden_layers = 2
	num_neurons = [200, 150]
	neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=False)
	neural_network.get_keras_model().fit(training_data, training_labels, batch_size=1)

	num_dme_pulses = 0
	correct_pulse_predictions = 0
	for i in range(len(validation_data)):
		validation_sample = validation_data[i]
		print(np.sum(validation_sample))
		predictions = neural_network.get_keras_model().predict(np.reshape(validation_sample, (1, sample_length, num_channels)))
		for j in range(0, len(validation_sample)):
			current_observation = validation_sample[j]
			current_label = validation_labels[i,j]
			current_prediction = np.rint(predictions[0,j,:])
			if int(np.sum(current_label)) == 2:
				num_dme_pulses = num_dme_pulses + 1
				print(current_observation, end=" -> ")
				print(current_prediction, end=" should be ")
				print(current_label, end=" ")
				if (current_prediction == current_label).all():
					print("✔")
					correct_pulse_predictions = correct_pulse_predictions + 1
				else:
					print("⨯")

	print(str(correct_pulse_predictions / num_dme_pulses * 100) + "% correct.")


def plot_online_learning():
	"""
	:return: _imgs/dme/lstm_online_learning.pdf
	"""
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 7  # send a DME request every x ms
	training_time = 2500
	num_repetitions = 12
	batch_means_split = 4
	sample_length = dme_request_frequency  # milliseconds
	validation_time = 1000
	simtime_max = training_time + validation_time
	util.verbose_print.verbose = False

	x_gs = 260*1000
	y_gs = 150*1000
	# 0.5ms propagation delay to ground station
	x_as_1 = 260*1000
	y_as_1 = 0
	# 0.5ms propagation delay to ground station
	x_dme_1 = 260*1000
	y_dme_1 = 300*1000
	# 0.25ms propagation delay to ground station
	x_dme_2 = 260*1000
	y_dme_2 = int(225*1000)
	# 1ms propagation delay to ground station
	x_as_2 = 260*1000
	y_as_2 = int(150*1000*3)

	ground_station = DMEGroundStation(0, 1, x_gs, y_gs, None)
	channel = InteractiveChannelModel(num_channels, ground_station, simtime_max)

	aircraft = Aircraft(x_as_1, y_as_1, None)
	prop_delay = channel.__get_propagation_delay__(channel.__euclidean_distance__(ground_station, aircraft))
	assert(prop_delay == 0.5)
	aircraft = Aircraft(x_dme_1, y_dme_1, None)
	prop_delay = channel.__get_propagation_delay__(channel.__euclidean_distance__(ground_station, aircraft))
	assert(prop_delay == 0.5)
	aircraft = Aircraft(x_dme_2, y_dme_2, None)
	prop_delay = channel.__get_propagation_delay__(channel.__euclidean_distance__(ground_station, aircraft))
	assert(prop_delay == 0.25)
	aircraft = Aircraft(x_as_2, y_as_2, None)
	prop_delay = channel.__get_propagation_delay__(channel.__euclidean_distance__(ground_station, aircraft))
	assert(prop_delay == 1.0)

	# Observations for first position.
	training_data1, training_labels1, _, _ = get_training_data(x_as_1, y_as_1, [800, 1], x_gs, y_gs, x_dme_1, y_dme_1, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length)
	# ... and for second position.
	training_data2, training_labels2, _, _ = get_training_data(x_as_2, y_as_2, [800, 1], x_gs, y_gs, x_dme_2, y_dme_2, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length)
	# their sum is all the samples that we have.
	num_training_samples = training_data1.shape[0] + training_data2.shape[0]

	plt.rcParams.update({'font.size': 28})
	plt.ylabel('channel index')
	plt.xlabel('time slot')
	observations = []
	for sample in range(2, 6):
		for i in range(sample_length):
			observations.append(training_data1[sample, i, :])
	observations = shift(observations, 1, 0)  # shift by one to have 1-indexing instead of 0-indexing
	observations_row = np.transpose(observations)
	plt.imshow(observations_row, cmap='Greys')
	plt.xticks(range(6, 5*6, 6))
	filename = "_imgs/dme/channel_access_pattern_1.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 4), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)

	plt.rcParams.update({'font.size': 28})
	plt.ylabel('channel index')
	plt.xlabel('time slot')
	observations = []
	for sample in range(2, 6):
		for i in range(sample_length):
			observations.append(training_data2[sample, i, :])
	observations = shift(observations, 1, 0)  # shift by one to have 1-indexing instead of 0-indexing
	observations_row = np.transpose(observations)
	plt.imshow(np.transpose(observations), cmap='Greys')
	plt.xticks(range(6, 5*6, 6))
	filename = "_imgs/dme/channel_access_pattern_2.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 4), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)

	learning_rate = 0.0005
	num_hidden_layers = 2
	num_neurons = [200, 150]

	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Sample [#]')
	plt.ylabel('Validation Accuracy')

	accuracy_mat = np.zeros((num_repetitions, num_training_samples))
	for rep in range(num_repetitions):
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=False)
		accuracy_vec1 = BinaryAccuracyHistory()
		print("rep " + str(rep+1) + " position 1")
		neural_network.get_keras_model().fit(training_data1, training_labels1, batch_size=1, callbacks=[accuracy_vec1])
		accuracy_vec2 = BinaryAccuracyHistory()
		print("rep " + str(rep+1) + " position 2")
		neural_network.get_keras_model().fit(training_data2, training_labels2, batch_size=1, callbacks=[accuracy_vec2])
		accuracy_mat[rep] = np.concatenate((accuracy_vec1.accuracies, accuracy_vec2.accuracies))

	# Compute batch-means for every data point.
	batch_means = columnwise_batch_means(accuracy_mat, batch_means_split)
	# Compute range for each data point using confidence intervals.
	sample_means = np.zeros(num_training_samples)
	sample_means_minus = np.zeros(num_training_samples)
	sample_means_plus = np.zeros(num_training_samples)
	confidence = 0.95
	for data_point in range(num_training_samples):
		sample_means[data_point], sample_means_minus[data_point], sample_means_plus[data_point] = calculate_confidence_interval(batch_means[:,data_point], confidence)

	x = range(len(sample_means))

	plt.plot(x, sample_means)
	plt.fill_between(x, sample_means_minus, sample_means_plus, alpha=0.5)
	plt.axvline(x=int(len(x)/2), color='black', alpha=0.75, linestyle='--', label="position change event")
	plt.legend()
	plt.ylim((0.5, 1))
	filename = "_imgs/dme/lstm_online_learning.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 10), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


def compare_sample_lengths():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5  # send a DME request every x ms
	training_time = 2500
	sample_lengths = [1, 2]  # milliseconds
	validation_time = 2500
	simtime_max = training_time + validation_time
	util.verbose_print.verbose = False

	x_gs = 260*1000
	y_gs = 150*1000
	# 0.5ms propagation delay to ground station
	x_as_1 = 260*1000
	y_as_1 = 0
	# 0.5ms propagation delay to ground station
	x_dme_1 = 260*1000
	y_dme_1 = 300*1000

	learning_rate = 0.0005
	num_hidden_layers = 2
	num_neurons = [200, 150]
	num_repetitions = 1

	accuracy_mat = np.zeros((len(sample_lengths), 3))
	confidence = 0.95

	for i in range(len(sample_lengths)):
		validation_accuracy_vec = np.zeros(num_repetitions)
		sample_length = sample_lengths[i]
		training_data, training_labels, validation_data, validation_labels = get_training_data(x_as_1, y_as_1, [800, 1], x_gs, y_gs, x_dme_1, y_dme_1, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_length)
		for rep in range(num_repetitions):
			print("length " + str(i+1) + "/" + str(len(sample_lengths)) + " rep " + str(rep+1) + "/" + str(num_repetitions))
			neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers, use_softmax=False)
			neural_network.get_keras_model().fit(training_data, training_labels, batch_size=1, shuffle=False, epochs=5)
			result_vec = neural_network.get_keras_model().evaluate(validation_data, validation_labels, batch_size=1)
			validation_accuracy_vec[rep] = result_vec[1]  # index 0 is loss, index 1 is accuracy
		accuracy_mat[i][0], accuracy_mat[i][1], accuracy_mat[i][2] = calculate_confidence_interval(validation_accuracy_vec, confidence)

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Validation Accuracy')
	plt.xlabel('Input Sequence Length')
	plt.errorbar(sample_lengths, accuracy_mat[:, 0], accuracy_mat[:, 2] - accuracy_mat[:, 0], fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
	plt.xticks(sample_lengths)

	filename = "_imgs/dme/lstm/lstm_accuracy_over_sample_lengths.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 10), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


def try_stateful_stuff():
	num_channels = 2
	interrogator_channel_index = 0
	response_channel_index = 1
	dme_request_frequency = 5  # send a DME request every x ms
	training_time = 10000
	sample_lengths = [1, 2]  # milliseconds
	validation_time = 500
	simtime_max = training_time + validation_time
	util.verbose_print.verbose = False

	x_gs = 260*1000
	y_gs = 150*1000
	# 0.5ms propagation delay to ground station
	x_as_1 = 260*1000
	y_as_1 = 0
	# 0.5ms propagation delay to ground station
	x_dme_1 = 260*1000
	y_dme_1 = 300*1000

	learning_rate = 0.0005
	num_hidden_layers = 2
	num_neurons = [200, 150]
	num_repetitions = 2

	accuracy_mat = np.zeros((len(sample_lengths), 3))
	confidence = 0.95

	training_data, training_labels, validation_data, validation_labels = get_training_data(x_as_1, y_as_1, [800, 1], x_gs, y_gs, x_dme_1, y_dme_1, [800, -1], interrogator_channel_index, response_channel_index, dme_request_frequency, training_time, validation_time, num_channels, sample_lengths[0])

	print(training_data.shape[1])

	n_batch = len(training_data)

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.LSTM(units=num_neurons[0], batch_input_shape=(1, 1, 2), stateful=True, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
	model.add(tf.keras.layers.Dense(units=num_channels, name="output_layer"))
	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0005), loss='mean_squared_error', metrics=['binary_accuracy'])
	model.fit(x=training_data, y=training_labels, shuffle=False, batch_size=1)

if __name__ == '__main__':
	# plot_learning_two_signals()
	# verify_prediction()
	plot_online_learning()  # _imgs/dme/lstm_online_learning.pdf
	# validate_training_data()
	# validate_training_data_sample_length_1()
	# compare_sample_lengths()
	# try_stateful_stuff()