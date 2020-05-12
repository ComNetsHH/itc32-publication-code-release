import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from predictor.neural_network import TumuluruMLPAdam
from environment.data_generator import *
import numpy as np
from util.callback_loss_history import *
import matplotlib.pyplot as plt
from util.confidence_intervals import *


def run_repeatedly_with_validation(sample_length, num_channels, activation_pattern, switching_prob, sensing_error_prob, number_of_timeslots, num_repetitions, confidence, batch_size, validation_timeslots):
	"""
	:param sample_length: Size of the input vector for the neural network.
	:param num_channels: Number of channels in the sequential access model.
	:param activation_pattern: The pattern through which the channel access model cycles.
	:param switching_prob: Probability of switching to the next channel.
	:param sensing_error_prob: Probability of wrongly sensing a channel.
	:param number_of_timeslots: The number of timeslots the model should be evaluated for. This divided by sample size gives the number of samples.
	:param num_repetitions: Number of repetitions.
	:param confidence: Confidence interval's confidence value.
	:param batch_size: Number of samples passed into the neural network in one iteration; i.e. 1 for 'online training', any number < all samples for mini-batch training
	:param use_adam: Whether to use the Adam optimizer, or Stochastic Gradient Descent.
	:param validation_timeslots: Number of timeslots to observe the channel for to generate a validation data set.
	:return: Mean accuracy during validation with confidence interval margins.
	"""
	mean_accuracy_vec = np.zeros(num_repetitions)

	for rep in range(num_repetitions):
		print("repetition " + str(rep+1) + "/" + str(num_repetitions))
		channel = ErroneousSequentialAccessChannelModel(num_channels, switching_prob, activation_pattern, sensing_error_prob)
		neural_network = TumuluruMLPAdam(lookback_length=sample_length)
		data_generator = DataGenerator(num_channels, sample_length, channel)
		num_samples = int(number_of_timeslots / sample_length)
		training_data = np.zeros((num_samples, sample_length))
		training_labels = np.zeros((num_samples, 1))
		for i in range(num_samples):
			data, label = data_generator.read_next(sample_length)
			training_data[i] = data[:, 0]  # MLP works on only one channel, so ignore everything about the other ones.
			training_labels[i] = label[-1, 0]

		neural_network.get_keras_model().fit(x=training_data, y=training_labels, batch_size=batch_size)  # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
		validation_data = np.zeros((num_samples, sample_length))
		validation_labels = np.zeros((num_samples, 1))
		for i in range(num_samples):
			data, label = data_generator.read_next(sample_length)
			validation_data[i] = data[:, 0]  # MLP works on only one channel, so ignore everything about the other ones.
			validation_labels[i] = label[-1, 0]
		_, mean_accuracy_vec[rep] = neural_network.get_keras_model().evaluate(x=validation_data, y=validation_labels, batch_size=batch_size)

	sample_mean, sample_mean_minus, sample_mean_plus = calculate_confidence_interval(mean_accuracy_vec, confidence)
	return sample_mean, sample_mean_minus, sample_mean_plus, neural_network


def plot_validation_sure_switching_two_channels():
	"""
	:return: _imgs/sequential/mlp_validation_over_switching_probs_two_channels.pdf
	"""
	sample_length = 1
	num_channels = 2
	activation_pattern = [0, 1]
	sensing_error_prob = 0.0
	number_of_timeslots = 2500
	num_repetitions = 5
	confidence = 0.95
	batch_size = 1
	validation_timeslots = 2500

	switching_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	y_vec = np.zeros(len(switching_probs))
	ym_vec = np.zeros(len(switching_probs))
	yp_vec = np.zeros(len(switching_probs))
	for i in range(len(switching_probs)):
		switching_prob = switching_probs[i]
		print("switching_prob " + str(i+1) + "/" + str(len(switching_probs)))
		y_vec[i], ym_vec[i], yp_vec[i], _ = run_repeatedly_with_validation(sample_length, num_channels, activation_pattern, switching_prob, sensing_error_prob, number_of_timeslots, num_repetitions, confidence, batch_size, validation_timeslots)

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Validation Accuracy')
	plt.xlabel('Switching Probability')
	# plt.plot(switching_probs, y_vec)
	# plt.fill_between(switching_probs, ym_vec, yp_vec, alpha=0.5)
	plt.errorbar(switching_probs, y_vec, yerr=yp_vec - y_vec)
	filename = "_imgs/sequential/mlp_validation_over_switching_probs_two_channels.pdf"
	fig = plt.gcf()
	fig.set_size_inches((12, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)


def plot_validation_sure_switching_three_channels():
	"""
	:return: _imgs/sequential/mlp_validation_over_switching_probs_three_channels.pdf
	"""
	sample_length = 1
	num_channels = 3
	activation_pattern = [0, 1, 2]
	sensing_error_prob = 0.0
	number_of_timeslots = 2500
	num_repetitions = 5
	confidence = 0.95
	batch_size = 1
	validation_timeslots = 2500

	switching_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	y_vec = np.zeros(len(switching_probs))
	ym_vec = np.zeros(len(switching_probs))
	yp_vec = np.zeros(len(switching_probs))
	for i in range(len(switching_probs)):
		switching_prob = switching_probs[i]
		print("switching_prob " + str(i+1) + "/" + str(len(switching_probs)))
		y_vec[i], ym_vec[i], yp_vec[i], _ = run_repeatedly_with_validation(sample_length, num_channels, activation_pattern, switching_prob, sensing_error_prob, number_of_timeslots, num_repetitions, confidence, batch_size, validation_timeslots)

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Validation Accuracy [%]')
	plt.xlabel('Switching Probability')
	plt.errorbar(switching_probs, y_vec, yerr=yp_vec - y_vec)
	filename = "_imgs/sequential/mlp_validation_over_switching_probs_three_channels.pdf"
	fig = plt.gcf()
	fig.set_size_inches((15, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)


def prediction_on_idle_channel_with_three_channels():
	"""
	:return: _imgs/sequential/mlp_prediction_over_input_three_channels.pdf
	"""
	sample_length = 1
	num_channels = 3
	activation_pattern = [0, 1, 2]
	sensing_error_prob = 0.0
	number_of_timeslots = 2500
	num_repetitions = 5
	confidence = 0.95
	batch_size = 1
	validation_timeslots = 1  # We're only interested in the trained model, so skip validation.

	input_data_idle = [0]
	input_data_busy = [1]
	predictions_on_idle = np.zeros(num_repetitions)
	predictions_on_busy = np.zeros(num_repetitions)

	switching_prob = 1
	for i in range(num_repetitions):
		_, _, _, neural_network = run_repeatedly_with_validation(sample_length, num_channels, activation_pattern, switching_prob, sensing_error_prob, number_of_timeslots, num_repetitions, confidence, batch_size, validation_timeslots)
		predictions_on_idle[i] = neural_network.get_keras_model().predict(x=input_data_idle)
		predictions_on_busy[i] = neural_network.get_keras_model().predict(x=input_data_busy)

	y_idle, ym_idle, yp_idle = calculate_confidence_interval(predictions_on_idle, confidence)
	y_busy, ym_busy, yp_busy = calculate_confidence_interval(predictions_on_busy, confidence)

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Prediction value $h_\\Theta{}(x)$')
	plt.xlabel('Input $x$')
	plt.errorbar([0, 1], [y_idle, y_busy], yerr=[yp_idle - y_idle, yp_busy - y_busy], fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
	plt.xticks([0, 1])
	filename = "_imgs/sequential/mlp_prediction_over_input_three_channels.pdf"
	fig = plt.gcf()
	fig.set_size_inches((13, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)


def accuracy_on_three_channels_over_input_lenghts():
	"""
	:return: _imgs/sequential/mlp_validation_of_three_channels_over_input_lengths.pdf
	"""
	input_sequence_lengths = range(1, 7)
	num_channels = 3
	activation_pattern = [0, 1, 2]
	switching_prob = 1
	sensing_error_prob = 0.0
	number_of_timeslots = 2500
	num_repetitions = 1
	confidence = 0.95
	batch_size = 1
	validation_timeslots = number_of_timeslots

	means = np.zeros(len(input_sequence_lengths))
	means_minus = np.zeros(len(input_sequence_lengths))
	means_plus = np.zeros(len(input_sequence_lengths))
	for i in range(len(input_sequence_lengths)):
		means[i], means_minus[i], means_plus[i], _ = run_repeatedly_with_validation(input_sequence_lengths[i], num_channels, activation_pattern, switching_prob, sensing_error_prob, number_of_timeslots, num_repetitions, confidence, batch_size, validation_timeslots)

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Accuracy [%]')
	plt.xlabel('Input Sequence Length')
	plt.errorbar(input_sequence_lengths, means, yerr=means_plus - means, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
	filename = "_imgs/sequential/mlp_validation_of_three_channels_over_input_lengths.pdf"
	fig = plt.gcf()
	fig.set_size_inches((13, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)


if __name__ == '__main__':
	plot_validation_sure_switching_two_channels()  # "_imgs/sequential/mlp_validation_over_switching_probs_two_channels.pdf"
	plot_validation_sure_switching_three_channels()  # "_imgs/sequential/mlp_validation_over_switching_probs_three_channels.pdf"
	prediction_on_idle_channel_with_three_channels()  # "_imgs/sequential/mlp_prediction_over_input_three_channels.pdf"
	accuracy_on_three_channels_over_input_lenghts()  # "_imgs/sequential/mlp_validation_of_three_channels_over_input_lengths.pdf"
