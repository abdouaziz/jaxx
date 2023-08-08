
## import packages
import jax
import jax.numpy as jnp
import os
import requests

## setup JAX to use TPUs if available
try:
    url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
    resp = requests.post(url)
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
except:
    pass

jax.devices()

#Create a matrix with values [[10, 1, 24], [20, 15, 14]] and assign it to `data`


data = jnp.array([[10, 1, 24], [20, 15, 14]])

#print(f"Data: {data}")


#Assign the transpose of `data` to `dataT` 

dataT= jnp.transpose(data)  #dataT = data.T
#print(f"DataT: {dataT}")
#Assign the element of `data` at index [0, 2] to `value` 

value = data[0,2]
#print(f"Value: {value}")

#Update the value of `data` at index [1, 1] to `100` 

data = data.at[1,1].set(100)
#print("Data at [1,1] update: ", data)
#Add `41` to the value of `data` at index [0, 0] 

data = data.at[0,0].add(41)
#print(f"Data at [0,0] add 41: {data}")

#Calculate the minimum values over axis=1 and assign it to `mins` 

mins = jnp.min(data, axis=1)
#print(f"mins: {mins}")
#Select the first row of values of `data` and assign it to `data_select` 

data_select = data[0,:]
#print(f"Data select: {data_select}")
#Append the row `data_select` to `data` 

data = jnp.vstack([data, data_select])
#print(f"Data append: {data}")
#Multiply the matrices `data` and `dataT` and assign it to `data_prod` 

data_prod = jnp.matmul(data, dataT)
#print(f"Data prod: {data_prod}")

#Convert the dtype of `data_prod` to `float32` 

data_prod = data_prod.astype(jnp.float32)
#print(f"Data prod: {data_prod}")
