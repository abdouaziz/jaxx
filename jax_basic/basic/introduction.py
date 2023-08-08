
 
#Exercise 1: Install the `jax` package

# Install the `jax` package using pip install jax

#Import the `jax` package**"""

import jax 

# Display the version of `jax`

print(jax.__version__)

# Display the default backend of `jax`

print(jax.default_backend())

#Exercise 5: Display the devices of the backend 

print(jax.devices())

#Create a JAX DeviceArray with values [10, 1, 24] and assign it to `data` 

data = jax.numpy.array([10, 1, 24])

#Display the type of `data` 

print(type(data))

#Display the shape of `data` 

print(data.shape)

#Transfer `data` to host and assign it to `data_host`** 

data_host = data.copy()

#Transfer `data_host` to device and assign it to `data_device`** 


data_devie = jax.device_put(data_host)
 