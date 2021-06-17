import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_dataset(f, x_start, x_end, num_points):
  # Generate a set of `num_points` linearly spaced points from `x_start` to `x_end`
  x = np.linspace(x_start, x_end, num_points)

  # Evaluate the function `f` for each point in the set x
  y = [ f(i) for i in x ]

  return np.array(x), np.array(y)

# Use lambda to define a function that can be assigned to a variable.
quadratic = lambda x : pow(x, 2)
x_start = -10
x_end = 10
num_points = 100

x, y = generate_dataset(quadratic, x_start, x_end, num_points)
#plt.plot(x, y, 'ro')
#plt.show()

def build_model(hidden_layer_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation="sigmoid"),
    tf.keras.layers.Dense(1)
  ])
  model.build(input_shape=(1,1))
  return model

hidden_layer_size = 50
model = build_model(hidden_layer_size)

# Check the model summary
model.summary()

# Check evaluating the model for a sample x value.
x_sample = np.array([1]).reshape(1,1)
print(model(x_sample))

learning_rate = 1e-3

optimizer = tf.keras.optimizers.SGD(learning_rate)

def compute_loss(y_true, y_pred):
  return tf.keras.losses.mean_squared_error(y_true, y_pred)

def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = compute_loss(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

num_iterations = 5000

def get_batch(x, y, size):
  i = np.random.choice(size)
  return x[i].reshape(1,1), y[i]

for i in range(num_iterations):
  x_batch, y_batch = get_batch(x, y, num_points)
  train_step(x_batch, y_batch)

def evaluate_function(x):
  return [ model(i.reshape(1,1))[0] for i in x ]

x_new = [ i + 0.3 for i in x ]
y_pred = evaluate_function(x_new)

plt.plot(x, y, 'ro')
plt.plot(x_new, y_pred, 'ro', color='b')
plt.show()
