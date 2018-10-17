# Madeleine Hall. CID 01403898.
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data
def load_space_csv_data(file_name):
    df = pd.read_csv(file_name, delim_whitespace=True)
    cols = list(df.columns.values)
    return df, cols

df, cols = load_space_csv_data('poverty.txt')

PovPct = df['PovPct'].values # this line converts dataframe columns into np arrays, as required
Brth15to17 = df['Brth15to17'].values
ViolCrime = df['ViolCrime'].values


# create tensorflow placeholders and graph
A = tf.placeholder(tf.float32, shape=(len(df),None), name='data')
b = tf.placeholder(tf.float32, shape=(len(df),1), name='target')
normal_eqn = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(A),A)),tf.matmul(tf.transpose(A),b)) # this is the tensorflow graph


# Regress Brth15to17 against PovPct
feed_dict = {A: np.transpose(np.array([PovPct,np.ones(len(df))])), b: np.transpose(np.array([Brth15to17]))}
with tf.Session() as sess:
    output = sess.run(normal_eqn, feed_dict=feed_dict)
# Report the equation expressing the solution. Plot the data and the solution and include as an image file.
print('\n Brth15to17 = ',output[0],'PovPct + ',output[1])
fig, ax = plt.subplots(1)
ax.scatter(PovPct,Brth15to17)
xlim = ax.get_xlim()
ax.plot([xlim[0],xlim[1]],[xlim[0]*output[0]+output[1],xlim[1]*output[0]+output[1]],c=(1,0,0))
plt.xlabel('PovPct')
plt.ylabel('Brth15to17')
plt.legend(['linear regression solution','state'])
plt.title('poverty level and teen birth rate in the US')
plt.savefig('assignment1.png')
plt.show()


# Regress Brth15to17 against PovPct and ViolCrime
feed_dict = {A: np.transpose(np.array([PovPct,ViolCrime,np.ones(len(df))])), b: np.transpose(np.array([Brth15to17]))}
with tf.Session() as sess:
    output = sess.run(normal_eqn, feed_dict=feed_dict)
# Report the equation expressing the solution
print('\n Brth15to17 = ',output[0],'PovPct + ',output[1],'ViolCrime + ',output[2])
