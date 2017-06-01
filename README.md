# TensorFlow Models
A collection of helpful models for streamlineing the creation of machine learning models.

## NeuralNet
A model for creating a fully connected feed forward neural net given a list of layer sizes and activation functions. For example
```python
model = models.NeuralNet.NeuralNet(
	[num_input, 100, num_output],
	[tf.nn.relu, tf.nn.sigmoid]
)
```
