import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from environment.data_generator import *
from predictor.neural_network import *
from util.callback_loss_history import LossHistory, AccuracyHistory
import matplotlib.pyplot as plt
import numpy as np
from util.confidence_intervals import *


def run_every_timeslot(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length, learning_rate, time_limit, num_rep, split, confidence):
	loss_mat = np.zeros((num_rep, time_limit))
	for rep in range(num_rep):
		print(str(rep+1) + " / " + str(num_rep))
		data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)
		loss_vec = np.zeros(time_limit)

		observation_mat = np.zeros((sample_length, num_channels))
		label_mat = np.zeros((sample_length, num_channels))

		for timeslot in range(time_limit):
			# Read next observation.
			observation_mat[timeslot % sample_length], label_mat[timeslot % sample_length] = data_generator.read_next(1)
			if timeslot % sample_length == 0:  # when history_size > 1 this ensures that training only happens when enough observations have been aggregated
				reshaped_input_matrix = np.reshape(observation_mat, (1, sample_length, num_channels))
				reshaped_label_matrix = np.reshape(label_mat, (1, sample_length, num_channels))
				# .fit is called not for the entire data, but for each input, so that the loss for every timeslot can be obtained
				history = LossHistory()
				neural_network.get_keras_model().fit(reshaped_input_matrix, reshaped_label_matrix, callbacks=[history])
				loss_vec[timeslot] = history.losses[0]
		loss_mat[rep] = loss_vec

	# Compute batch-means for every data point.
	batch_means = columnwise_batch_means(loss_mat, split)
	# Compute range for each data point using confidence intervals.
	sample_means = np.zeros(time_limit)
	sample_means_minus = np.zeros(time_limit)
	sample_means_plus = np.zeros(time_limit)
	for data_point in range(time_limit):
		sample_means[data_point], sample_means_minus[data_point], sample_means_plus[data_point] = calculate_confidence_interval(batch_means[:,data_point], confidence)

	x = range(1, time_limit+1)
	return x, sample_means, sample_means_minus, sample_means_plus


def run_with_validation(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length, learning_rate, num_training_samples, num_validation_samples, num_rep, confidence, batch_size):
	mean_accuracy_vec = np.zeros(num_rep)
	neural_network = None

	for rep in range(num_rep):
		print("- repetition " + str(rep+1) + "/" + str(num_rep) + " -")
		data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
		neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)

		# Generate training data.
		observation_mat, labels_mat = data_generator.read_next(num_training_samples)  # num_samples x num_channel matrices
		# reshape to batch x time_step x data
		observation_mat = np.reshape(observation_mat, (num_training_samples, sample_length, num_channels))
		labels_mat = np.reshape(labels_mat, (num_training_samples, sample_length, num_channels))
		# Train.
		neural_network.get_keras_model().fit(observation_mat, labels_mat)

		# Generate validation data.
		validation_mat, validation_labels = data_generator.read_next(num_validation_samples)
		validation_mat = np.reshape(validation_mat, (num_validation_samples, sample_length, num_channels))
		validation_labels = np.reshape(validation_labels, (num_validation_samples, sample_length, num_channels))
		# Validate.
		_, mean_accuracy_vec[rep] = neural_network.get_keras_model().evaluate(x=validation_mat, y=validation_labels)

	sample_mean, sample_mean_minus, sample_mean_plus = calculate_confidence_interval(mean_accuracy_vec, confidence)
	return sample_mean, sample_mean_minus, sample_mean_plus, neural_network


def plot_lstm_loss_per_timeslot():
	"""
	:return: _imgs/sequential/lstm_loss_per_timeslot.pdf
	"""
	num_channels = 16
	activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	switching_prob = 1.0
	sample_length = 1
	learning_rate = 0.005
	time_limit = 10 * num_channels
	num_hidden_layers = 1
	num_neurons = [200]

	num_rep = 40
	split = int(num_rep / 5)
	confidence = 0.95

	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Timeslot')
	for x in range(num_channels, time_limit+1, num_channels):
		if x == num_channels:
			plt.axvline(x=x, color='lightgray', label='end of full pattern')
		else:
			plt.axvline(x=x, color='lightgray')
	plt.plot(range(time_limit), [1/num_channels]*time_limit, color='black', label='random guessing')

	x, y, y_m, y_p = run_every_timeslot(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length, learning_rate, time_limit, num_rep, split, confidence)
	plt.plot(x, y, label="LSTM Loss")
	plt.fill_between(x, y_m, y_p, alpha=0.5)
	plt.legend()

	filename = "_imgs/sequential/lstm_loss_per_timeslot.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 9), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


# def plot_accuracy_on_perfect_access_model():
# 	num_channels = 16
# 	activation_pattern = range(0, num_channels)
# 	switching_prob = 1.0
# 	sample_length = range(1, num_channels + 2)
# 	learning_rate = 0.005
# 	training_time_limit = 50 * num_channels
# 	validation_time_limit = 1000 * num_channels
# 	batch_size = 1
# 	num_hidden_layers = 1
# 	num_neurons = [150]
#
# 	num_rep = 20
# 	confidence = 0.95
#
# 	plt.rcParams.update({'font.size': 32})
# 	plt.xlabel('Input Sequence Length $T$')
# 	plt.ylabel('Validation Accuracy')
# 	yvec = np.zeros(len(sample_length))
# 	ypvec = np.zeros(len(sample_length))
# 	ymvec = np.zeros(len(sample_length))
# 	for i in range(len(sample_length)):
# 		yvec[i], ymvec[i], ypvec[i], _ = run_with_validation(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length[i], learning_rate, training_time_limit, validation_time_limit, num_rep, confidence, batch_size)
# 	plt.errorbar(sample_length, yvec, ypvec - yvec, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label="$p=" + str(switching_prob) + "$")
# 	plt.xticks(sample_length)
#
# 	filename = "_imgs/sequential-access/lstm/lstm_accuracy_perfect_model.pdf"
# 	fig = plt.gcf()
# 	fig.set_size_inches((16, 10), forward=False)
# 	fig.savefig(filename, dpi=500)
# 	plt.close()
# 	print("Graph saved to " + filename)
#
#
# def plot_accuracy_on_imperfect_access_model():
# 	num_channels = 16
# 	activation_pattern = range(0, num_channels)
# 	switching_prob = 0.75
# 	sample_length = range(1, num_channels + 2)
# 	learning_rate = 0.005
# 	training_time_limit = 50 * num_channels
# 	validation_time_limit = 1000 * num_channels
# 	batch_size = 1
# 	num_hidden_layers = 1
# 	num_neurons = [150]
#
# 	num_rep = 20
# 	confidence = 0.95
#
# 	plt.rcParams.update({'font.size': 32})
# 	plt.xlabel('Input Sequence Length $T$')
# 	plt.ylabel('Validation Accuracy')
# 	yvec = np.zeros(len(sample_length))
# 	ypvec = np.zeros(len(sample_length))
# 	ymvec = np.zeros(len(sample_length))
# 	for i in range(len(sample_length)):
# 		yvec[i], ymvec[i], ypvec[i], _ = run_with_validation(num_neurons, num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length[i], learning_rate, training_time_limit, validation_time_limit, num_rep, confidence, batch_size)
# 	plt.errorbar(sample_length, yvec, ypvec - yvec, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label="$p=" + str(switching_prob) + "$")
# 	plt.xticks(sample_length)
#
# 	filename = "_imgs/sequential-access/lstm/lstm_accuracy_imperfect_model.pdf"
# 	fig = plt.gcf()
# 	fig.set_size_inches((16, 10), forward=False)
# 	fig.savefig(filename, dpi=500)
# 	plt.close()
# 	print("Graph saved to " + filename)
#
#
# def plot_accuracy_over_num_neurons():
# 	num_channels = 16
# 	activation_pattern = range(0, num_channels)
# 	switching_prob = 0.75
# 	sample_length = 1
# 	learning_rate = 0.005
# 	training_time_limit = 10 * num_channels
# 	validation_time_limit = 100 * num_channels
# 	batch_size = 1
# 	num_hidden_layers = 1
# 	num_neurons = []
# 	for neurons in range(50, 500, 50):
# 		num_neurons.append([neurons])
#
# 	num_rep = 20
# 	confidence = 0.95
#
# 	plt.rcParams.update({'font.size': 32})
# 	plt.xlabel('Input Sequence Length $T$')
# 	plt.ylabel('Validation Accuracy')
# 	yvec = np.zeros(len(num_neurons))
# 	ypvec = np.zeros(len(num_neurons))
# 	ymvec = np.zeros(len(num_neurons))
# 	for i in range(len(num_neurons)):
# 		yvec[i], ymvec[i], ypvec[i], _ = run_with_validation(num_neurons[i], num_hidden_layers, num_channels, switching_prob, activation_pattern, sample_length, learning_rate, training_time_limit, validation_time_limit, num_rep, confidence, batch_size)
# 	plt.errorbar(num_neurons, yvec, ypvec - yvec, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
# 	plt.xticks(num_neurons)
#
# 	filename = "_imgs/sequential-access/lstm/lstm_accuracy_over_num_neurons.pdf"
# 	fig = plt.gcf()
# 	fig.set_size_inches((16, 10), forward=False)
# 	fig.savefig(filename, dpi=500)
# 	plt.close()
# 	print("Graph saved to " + filename)
#
#
# def plot_accuracy_over_num_hidden_layers():
# 	num_channels = 16
# 	activation_pattern = range(0, num_channels)
# 	switching_prob = 0.75
# 	sample_length = 1
# 	learning_rate = 0.005
# 	training_time_limit = 10 * num_channels
# 	validation_time_limit = 100 * num_channels
# 	batch_size = 1
# 	num_hidden_layers = [1, 2, 3, 4, 5]
# 	num_neurons = [[200], [200, 200], [200, 200, 200], [200, 200, 200, 200], [200, 200, 200, 200, 200]]
#
# 	num_rep = 20
# 	confidence = 0.95
#
# 	plt.rcParams.update({'font.size': 32})
# 	plt.xlabel('Number of Hidden Dense Layers')
# 	plt.ylabel('Validation Accuracy')
# 	yvec = np.zeros(len(num_hidden_layers))
# 	ypvec = np.zeros(len(num_hidden_layers))
# 	ymvec = np.zeros(len(num_hidden_layers))
# 	for i in range(len(num_hidden_layers)):
# 		yvec[i], ymvec[i], ypvec[i], _ = run_with_validation(num_neurons[i], num_hidden_layers[i], num_channels, switching_prob, activation_pattern, sample_length, learning_rate, training_time_limit, validation_time_limit, num_rep, confidence, batch_size)
# 	plt.errorbar(num_hidden_layers, yvec, ypvec - yvec, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
# 	plt.xticks(num_hidden_layers)
#
# 	filename = "_imgs/sequential-access/lstm/lstm_accuracy_over_num_hidden_layers.pdf"
# 	fig = plt.gcf()
# 	fig.set_size_inches((16, 10), forward=False)
# 	fig.savefig(filename, dpi=500)
# 	plt.close()
# 	print("Graph saved to " + filename)


def plot_accuracy_over_switching_prob():
	num_channels = 16
	activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	switching_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	sample_length = 1
	learning_rate = 0.005
	training_time_limit = 100 * num_channels
	validation_time_limit = 200 * num_channels
	batch_size = 1
	num_hidden_layers = 1
	num_neurons = [150]

	num_rep = 10
	confidence = 0.95

	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Switching Probability')
	plt.ylabel('Validation Accuracy')
	yvec = np.zeros(len(switching_probs))
	ypvec = np.zeros(len(switching_probs))
	ymvec = np.zeros(len(switching_probs))
	for i in range(len(switching_probs)):
		print("--- switching prob " + str(i+1) + " / " + str(len(switching_probs)) + " ---")
		yvec[i], ymvec[i], ypvec[i], _ = run_with_validation(num_neurons, num_hidden_layers, num_channels, switching_probs[i], activation_pattern, sample_length, learning_rate, training_time_limit, validation_time_limit, num_rep, confidence, batch_size)
	# plt.errorbar(switching_probs, yvec, ypvec - yvec, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
	plt.errorbar(switching_probs, yvec, ypvec - yvec)
	plt.xticks(switching_probs)

	filename = "_imgs/sequential-access/lstm/lstm_accuracy_over_switching_probs.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 10), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


def plot_prediction_over_time():
	"""
	:return: _imgs/sequential/lstm_prediction_over_timeslots.pdf
	"""
	num_channels = 16
	activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	switching_prob = 0.75
	sample_length = 1
	learning_rate = 0.005
	num_training_samples = 5000 * num_channels
	num_hidden_layers = 1
	num_neurons = [200]

	# Get trained neural network.
	data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=True)
	neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)
	# Generate training data.
	observation_mat, labels_mat = data_generator.read_next(num_training_samples)  # num_samples x num_channel matrices
	# reshape to batch x time_step x data
	observation_mat = np.reshape(observation_mat, (num_training_samples, sample_length, num_channels))
	labels_mat = np.reshape(labels_mat, (num_training_samples, sample_length, num_channels))
	# Train.
	neural_network.get_keras_model().fit(observation_mat, labels_mat, shuffle=False)

	# For the next couple of pattern cycles
	validation_timeslots = 10*num_channels
	# Keep track of the prediction value for the first channel being idle...
	prediction_vec = np.zeros(validation_timeslots)
	first_channel_actually_idle = []
	for i in range(validation_timeslots):
		observation, label = data_generator.read_next(1)
		if np.argmax(observation) == 0:
			first_channel_actually_idle.append(i)
		observation = np.reshape(observation, (1, sample_length, num_channels))
		prediction = neural_network.get_keras_model().predict(x=observation)
		prediction_on_first_channel_being_idle = prediction[0,0,0]
		prediction_vec[i] = prediction_on_first_channel_being_idle

	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Timeslot [#]')
	plt.scatter(range(1, validation_timeslots+1), prediction_vec, label="$h_\Theta{}$(1st channel idle)")
	for i in range(len(first_channel_actually_idle)):
		if i==0:
			plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--', label='1st channel idle')
		else:
			plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--')
	plt.axhline(switching_prob, label="switching probability", color='black')
	plt.yticks([0, 0.25, 0.5, 0.75])
	plt.legend()
	filename = "_imgs/sequential/lstm_prediction_over_timeslots.pdf"
	fig = plt.gcf()
	fig.set_size_inches((16, 10), forward=False)
	fig.savefig(filename, dpi=500)
	plt.close()
	print("Graph saved to " + filename)


def plot_prediction_over_time_for_partially_observable_model():
	"""
	Assuming a partially observable model, where only the first frequency channel is observed in every timeslot,
	plot the prediction value of the first channel being idle over a number of timeslots on a trained model.
	:return:
	"""
	num_channels = 16
	activation_pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	switching_prob = 1
	sample_length = 1
	learning_rate = 0.005
	num_timeslots = 100 * num_channels
	num_hidden_layers = 1
	num_neurons = [150]

	data_generator = DataGenerator(num_channels, sample_length, SequentialAccessChannelModel(num_channels, switching_prob, activation_pattern), update_once=False)
	neural_network = LSTMNetwork(num_channels, num_neurons, learning_rate, sample_length, num_hidden_layers)
	initial_observation = np.zeros(num_channels)
	initial_observation = np.reshape(initial_observation, (1, sample_length, num_channels))

	prediction = neural_network.get_keras_model().predict(initial_observation)
	channel_selection = np.argmin(prediction)  # This is the channel that is picked for sensing/transmitting next.
	for i in range(num_timeslots):
		# if i % num_channels == 0:
		# 	print("\r" + str(i/num_channels+1) + " / " + str(num_timeslots/num_channels), end="")
		# Read full channel states.
		observation, label = data_generator.read_next(1)
		# Throw away all channel states of channels that were not selected.
		for j in range(num_channels):
			observation[0, j] = 0 if j != channel_selection else observation[0, j]
		observation = np.reshape(observation, (1, sample_length, num_channels))
		prediction = neural_network.get_keras_model().predict(observation)
		for j in range(num_channels):
			label[0, j] = prediction[0,0,j] if j != channel_selection else label[0, j]
		# print(label)

		label = np.reshape(label, (1, sample_length, num_channels))
		neural_network.get_keras_model().fit(x=observation, y=label, verbose=False)
		prediction = neural_network.get_keras_model().predict(observation)
		channel_selection = np.argmin(prediction)  # This is the channel that is picked for sensing/transmitting next.
		print(str(channel_selection) + " -> ", end="", flush=True)
	print()

	# For the next couple of pattern cycles
	validation_timeslots = 10*num_channels
	# Keep track of the prediction value for the first channel being idle...
	prediction_vec = np.zeros(validation_timeslots)
	first_channel_actually_idle = []
	for i in range(validation_timeslots):
		observation, label = data_generator.read_next(1)
		if np.argmax(observation) == 0:
			first_channel_actually_idle.append(i)
		observation = np.reshape(observation, (1, sample_length, num_channels))
		prediction = neural_network.get_keras_model().predict(x=observation)
		prediction_on_first_channel_being_idle = prediction[0,0,0]
		prediction_vec[i] = prediction_on_first_channel_being_idle

	plt.rcParams.update({'font.size': 32})
	plt.xlabel('Timeslot [#]')
	plt.scatter(range(1, validation_timeslots+1), prediction_vec, label="$h_\Theta{}$(1st channel idle)")
	for i in range(len(first_channel_actually_idle)):
		if i==0:
			plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--', label='1st channel idle')
		else:
			plt.axvline(first_channel_actually_idle[i], color='gray', alpha=0.5, linestyle='--')
	plt.axhline(switching_prob, label="switching probability", color='black')
	plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
	plt.legend()
	plt.show()

# filename = "_imgs/sequential-access/lstm/lstm_prediction_over_timeslots.pdf"
# fig = plt.gcf()
# fig.set_size_inches((16, 10), forward=False)
# fig.savefig(filename, dpi=500)
# plt.close()
# print("Graph saved to " + filename)


if __name__ == '__main__':
	plot_lstm_loss_per_timeslot()  # _imgs/sequential/lstm_loss_per_timeslot.pdf
	# plot_accuracy_on_perfect_access_model()
	# plot_accuracy_on_imperfect_access_model()
	# plot_accuracy_over_num_neurons()
	# plot_accuracy_over_num_hidden_layers()
	# plot_accuracy_over_switching_prob()
	plot_prediction_over_time()  # _imgs/sequential/lstm_prediction_over_timeslots.pdf
	# plot_prediction_over_time_for_partially_observable_model()