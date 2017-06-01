# TensorFlow Models
**tensorflow-models** - TensorFlow machine learning models.

## Purpose 
A collection of helpful models for streamlining the creation of machine learning models Developed in the CAMEL lab at Clarkson University for anomaly detection.

## Features
Currently, just contains the NeuralNet model. Planning to add more models.

# Usage

## NeuralNet
A model for creating a fully connected feed forward neural net given a list of layer sizes and activation functions.

### Initialization 
We can create a model by passing a list of layer sizes and a list of activation functions.
```python
model = models.NeuralNet.NeuralNet(
	[num_input, 100, num_output],
	[tf.nn.relu, tf.nn.sigmoid]
)
```

### Building the Model
We can build the computation graph using the function `create_network`. An example is provided below
```python
X = tf.placeholder('float', [None, num_inputs])
keep_prob = tf.placeholder('float')
prediction = model.create_network(X, keep_prob)
``` 

### L2 Loss for Objective Function
The L2 loss can be added the objective function by calling `get_l2_loss`.
```python
cost += 0.01 * model.get_l2_loss()
```

### Reset Weights
Using `reset_weights` returns a tensor operation to reset the weights using `sess.run`
```python
with tf.Session() as sess:
	sess.run(model.reset_weights())
```
