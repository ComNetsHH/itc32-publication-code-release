import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from predictor.neural_network import *
from environment.data_generator import *
import numpy as np
from util.callback_loss_history import *
import matplotlib.pyplot as plt
from util.confidence_intervals import *


def generate_data(channel, num_timeslots, sample_length):
	"""
	:param channel: The channel model.
	:param num_timeslots: Number of timeslots to generate.
	:param sample_length: Length of one input sample.
    :return: (num_timeslots/sample_length, sample_vec)-matrix.
	"""
	num_samples = int(num_timeslots / sample_length)
	data_mat = np.zeros((num_samples, sample_length))
	label_vec = np.zeros(num_samples)
	for sample in range(num_samples):
		for i in range(sample_length):
			channel.update()
			data_mat[sample][i] = channel.get_state_vector()[0]
			if i == 0 and sample > 0:
				label_vec[sample - 1] = channel.get_state_vector()[0]
	channel.update()
	label_vec[num_samples - 1] = channel.get_state_vector()[0]
	return data_mat, label_vec


def run_once(sample_length, mean_idle_slots, mean_busy_slots, number_of_timeslots):
	channel = PoissonProcessChannelModel(mean_idle_slots, mean_busy_slots)
	neural_network = TumuluruMLP(lookback_length=sample_length)
	training_data, training_labels = generate_data(channel, number_of_timeslots, sample_length)

	history = AccuracyHistory()
	neural_network.get_keras_model().fit(x=training_data, y=training_labels, callbacks=[history], batch_size=1)
	return range(1, len(history.accuracies) + 1), history.accuracies


def run_repeatedly(sample_length, mean_idle_slots, mean_busy_slots, number_of_timeslots, num_repetitions, confidence, batch_size, batch_means_split, use_adam):
	"""
	:param sample_length: Size of the input vector for the neural network.
	:param mean_idle_slots: Average number of consecutive timeslots that are idle.
	:param mean_busy_slots: Average number of consecutive timeslots that are busy.
	:param number_of_timeslots: The number of timeslots the model should be evaluated for. This divided by sample size gives the number of samples.
	:param num_repetitions: Number of repetitions.
	:param confidence: Confidence interval's confidence value.
	:param batch_size: Number of samples passed into the neural network in one iteration; i.e. 1 for 'online training', any number < all samples for mini-batch training
	:param batch_means_split: To apply batch-means, run for 'num_repetitions' repetitions, and then split these into 'batch_mean_splits' groups.
	:param use_adam: Whether to use the Adam optimizer, or Stochastic Gradient Descent.
	:return: Mean accuracy during training with confidence interval margins.
	"""
	num_samples = int(number_of_timeslots / sample_length)  # one sample has several items
	accuracy_history_mat = np.zeros((num_repetitions, num_samples))

	for rep in range(num_repetitions):
		channel = PoissonProcessChannelModel(mean_idle_slots, mean_busy_slots)
		neural_network = TumuluruMLPAdam(lookback_length=sample_length) if use_adam else TumuluruMLP(
			lookback_length=sample_length)
		training_data, training_labels = generate_data(channel, number_of_timeslots, sample_length)

		history = BinaryAccuracyHistory()
		neural_network.get_keras_model().fit(x=training_data, y=training_labels, callbacks=[history], batch_size=batch_size)  # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
		accuracy_history_mat[rep] = history.accuracies

	# Compute batch-means for every data point.
	batch_means = columnwise_batch_means(accuracy_history_mat, batch_means_split)  # rows are splits, columns are data point means over the respective splits
	# Compute range for each data point using confidence intervals.
	sample_means = np.zeros(num_samples)
	sample_means_minus = np.zeros(num_samples)
	sample_means_plus = np.zeros(num_samples)
	for data_point in range(num_samples):
		sample_means[data_point], sample_means_minus[data_point], sample_means_plus[
			data_point] = calculate_confidence_interval(batch_means[:, data_point], confidence)
	x = range(1, num_samples + 1)
	return x, sample_means, sample_means_minus, sample_means_plus


def run_repeatedly_with_validation(sample_length, mean_idle_slots, mean_busy_slots, number_of_timeslots, num_repetitions, confidence, batch_size, use_adam, validation_timeslots):
	"""
	:param sample_length: Size of the input vector for the neural network.
	:param mean_idle_slots: Average number of consecutive timeslots that are idle.
	:param mean_busy_slots: Average number of consecutive timeslots that are busy.
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
		channel = PoissonProcessChannelModel(mean_idle_slots, mean_busy_slots)
		neural_network = TumuluruMLPAdam(lookback_length=sample_length) if use_adam else TumuluruMLP(lookback_length=sample_length)
		training_data, training_labels = generate_data(channel, number_of_timeslots, sample_length)

		neural_network.get_keras_model().fit(x=training_data, y=training_labels, batch_size=batch_size)  # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
		validation_data, validation_labels = generate_data(channel, validation_timeslots, sample_length)
		_, mean_accuracy_vec[rep] = neural_network.get_keras_model().evaluate(x=validation_data, y=validation_labels, batch_size=batch_size)

	sample_mean, sample_mean_minus, sample_mean_plus = calculate_confidence_interval(mean_accuracy_vec, confidence)
	return sample_mean, sample_mean_minus, sample_mean_plus


def plot_training_phase():
	"""
	:return: _imgs/poisson/mlp_adam.pdf
	"""
	mean_idle_slots = 5  # Expectation value for Geometric distribution for idle timeslots
	mean_busy_slots = 10  # Expectation value for Geometric distribution for busy timeslots
	number_of_timeslots = 2500  # evaluate this many timeslots
	num_repetitions = 60
	batch_means_split = 5
	confidence = 0.95
	batch_size = 1
	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Training Accuracy')
	plt.xlabel('Sample [#]')
	colors = ['blue', 'orange']
	sample_sizes = [1, 4]
	xVec = []
	yVec = []
	ymVec = []
	ypVec = []
	for i in range(len(sample_sizes)):
		x, y, y_m, y_p = run_repeatedly(sample_sizes[i], mean_idle_slots, mean_busy_slots, number_of_timeslots, num_repetitions, confidence, batch_size, batch_means_split, use_adam=True)
		xVec.append(x)
		yVec.append(y)
		ymVec.append(y_m)
		ypVec.append(y_p)
	for i in range(len(sample_sizes)):
		plt.plot(xVec[i], yVec[i], colors[i], label="T=" + str(sample_sizes[i]))
		plt.fill_between(xVec[i], ymVec[i], ypVec[i], facecolor=colors[i], alpha=0.5)
	plt.legend()

	filename = "_imgs/poisson/mlp_adam.pdf"
	fig = plt.gcf()
	fig.set_size_inches((12, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)
	plt.close()


def plot_validation_accuracy_over_input_length_both():
	"""
	:return: _imgs/poisson/mlp_adam_validation_over_input_lengths_both.pdf
	"""
	mean_idle_slots = [2, 22]  # Expectation value for Geometric distribution for idle timeslots
	mean_busy_slots = [4, 46]  # Expectation value for Geometric distribution for busy timeslots
	number_of_timeslots = 2500
	validation_timeslots = 2500

	num_repetitions = 10
	confidence = 0.95
	batch_size = 1
	use_adam = True

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Validation Accuracy')
	plt.xlabel('Sample length $T$')
	sample_lengths = range(1, 17)  # 1..16
	means = np.zeros(len(sample_lengths))
	means_minus = np.zeros(len(sample_lengths))
	means_plus = np.zeros(len(sample_lengths))
	for j in range(len(mean_idle_slots)):
		idle_slots = mean_idle_slots[j]
		busy_slots = mean_busy_slots[j]
		for i in range(len(sample_lengths)):
			means[i], means_minus[i], means_plus[i] = run_repeatedly_with_validation(sample_lengths[i], idle_slots, busy_slots, number_of_timeslots, num_repetitions, confidence, batch_size, use_adam, validation_timeslots)
		plt.errorbar(sample_lengths, means, yerr=means_plus - means, fmt='o' if j == 0 else 'x', color='black', ecolor='lightblue' if j == 0 else "orange", elinewidth=3, capsize=0, label="short periods" if j == 0 else "long periods")
	plt.legend()
	plt.xticks(sample_lengths)

	filename = "_imgs/poisson/mlp_adam_validation_over_input_lengths_both.pdf"
	fig = plt.gcf()
	fig.set_size_inches((13, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)
	plt.close()


def plot_predictions_over_time(num_repetitions, batch_means_split, filename):
	"""
	:return: _imgs/poisson/mlp_adam_prediction_over_time.pdf
	"""
	mean_idle_slots = 3
	mean_busy_slots = 1
	channel = PoissonProcessChannelModel(mean_idle_slots, mean_busy_slots)
	print('rho=' + str(channel.get_utilization()))

	sample_length = 1  # We input a single observation into the neural network.
	neural_network = TumuluruMLPAdam(lookback_length=sample_length)
	num_timeslots = 2500
	num_samples = int(num_timeslots / sample_length)

	# Prepare idle channel input vector.
	input_vec_idle = np.zeros((1, sample_length))

	# And the same for a busy channel.
	input_vec_busy = np.zeros((1, sample_length))
	for i in range(len(input_vec_busy)):
		input_vec_busy[0, i] = 1

	predictions_idle = np.zeros((num_repetitions, num_samples))
	predictions_busy = np.zeros((num_repetitions, num_samples))
	for rep in range(num_repetitions):
		print(str(rep+1) + " / " + str(num_repetitions))
		# This keeps track of what the neural network predicts for an idle channel after every training batch.
		prediction_history_idle = PredictionHistory(neural_network, input_vec_idle)
		# And the prediction on a now-busy channel.
		prediction_history_busy = PredictionHistory(neural_network, input_vec_busy)

		training_data, training_labels = generate_data(channel, num_timeslots, sample_length)
		neural_network.get_keras_model().fit(x=training_data, y=training_labels, batch_size=1, callbacks=[prediction_history_idle, prediction_history_busy])
		predictions_idle[rep] = np.reshape(prediction_history_idle.predictions, len(prediction_history_idle.predictions))
		predictions_busy[rep] = np.reshape(prediction_history_busy.predictions, len(prediction_history_idle.predictions))
	# Collect batch means.
	idle_batch_means = columnwise_batch_means(predictions_idle, batch_means_split)
	busy_batch_means = columnwise_batch_means(predictions_busy, batch_means_split)

	# Calculate confidence intervals on the batch means.
	idle_ci_means = np.zeros(num_timeslots)
	idle_ci_minus = np.zeros(num_timeslots)
	idle_ci_plus = np.zeros(num_timeslots)
	busy_ci_means = np.zeros(num_timeslots)
	busy_ci_minus = np.zeros(num_timeslots)
	busy_ci_plus = np.zeros(num_timeslots)
	confidence = 0.95
	for timeslot in range(num_samples):
		idle_ci_means[timeslot], idle_ci_minus[timeslot], idle_ci_plus[timeslot] = calculate_confidence_interval(idle_batch_means[:,timeslot], confidence)
		busy_ci_means[timeslot], busy_ci_minus[timeslot], busy_ci_plus[timeslot] = calculate_confidence_interval(busy_batch_means[:,timeslot], confidence)
	x = range(len(idle_ci_means))

	plt.rcParams.update({'font.size': 32})
	plt.ylabel('Prediction value $h_\\Theta$')
	plt.xlabel('Training sample [#]')

	plt.plot(x, idle_ci_means, color='lightblue', label="$h_\\Theta{}(idle)$")
	plt.fill_between(x, idle_ci_minus, idle_ci_plus, facecolor='lightblue', alpha=0.25)

	plt.plot(x, busy_ci_means, color='orange', label="$h_\\Theta{}(busy)$")
	plt.fill_between(x, busy_ci_minus, busy_ci_plus, facecolor='orange', alpha=0.25)

	plt.axhline(y=1-channel.get_utilization(), color='k', linestyle='-', label="$1-\\rho$")
	plt.axhline(y=channel.get_utilization(), color='k', linestyle='--', label="$\\rho$")
	plt.ylim(0, 1)
	plt.legend()

	fig = plt.gcf()
	fig.set_size_inches((13, 10), forward=False)
	fig.savefig(filename, dpi=500)
	print("File saved to " + filename)
	plt.close()


if __name__ == '__main__':
	# plot_training_phase()  # _imgs/poisson/mlp_adam.pdf
	# plot_validation_accuracy_over_input_length_both()  # _imgs/poisson/mlp_adam_validation_over_input_lengths_both.pdf
	plot_predictions_over_time(1, 1, "_imgs/poisson/mlp_adam_prediction_over_time.pdf")  # _imgs/poisson/mlp_adam_prediction_over_time.pdf
	# plot_predictions_over_time(12, 3, "_imgs/poisson/mlp_adam_prediction_over_time-averages.pdf")  # _imgs/poisson/mlp_adam_prediction_over_time-averages.pdf
