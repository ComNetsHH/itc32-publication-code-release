import tensorflow.compat.v1 as tf

class LossHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


class AccuracyHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.accuracies = []

	def on_batch_end(self, batch, logs={}):
		self.accuracies.append(logs.get('accuracy'))


class BinaryAccuracyHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.accuracies = []

	def on_batch_end(self, batch, logs={}):
		self.accuracies.append(logs.get('binary_accuracy'))


class PredictionHistory(tf.keras.callbacks.Callback):
	def __init__(self, neural_network, input_vec):
		self.neural_network = neural_network
		self.input_vec = input_vec
		self.predictions = []

	def on_train_begin(self, logs={}):
		self.predictions = []

	def on_batch_end(self, batch, logs={}):
		self.predictions.append(self.neural_network.get_keras_model().predict(x=self.input_vec, batch_size=1))
