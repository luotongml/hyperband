
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:

import data_processing
path = "data/bigdf.pkl"
df = data_processing.load_options(path)
for data in  data_processing.train_test_split(df=df, window=1):
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]


# In[3]:

x_train.head()
x_train['moneyness'].describe()


# In[4]:

m_values = x_train['moneyness'].reshape((-1,1)).astype(np.float32)
t_values = x_train['years_to_exe'].reshape((-1,1)).astype(np.float32)
s_values = x_train['etf_mid'].reshape((-1,1)).astype(np.float32)
c_values = y_train.reshape((-1,1)).astype(np.float32)


# In[5]:

m_test_values = x_test['moneyness'].reshape((-1,1)).astype(np.float32)
t_test_values = x_test['years_to_exe'].reshape((-1,1)).astype(np.float32)
s_test_values = x_test['etf_mid'].reshape((-1,1)).astype(np.float32)
c_test_values = y_test.reshape((-1,1)).astype(np.float32)


# In[6]:




# In[6]:

num_features = 3
num_neurons = 5 #J
num_gates = 9 #I could be changed
num_gate_neurons = 5 #K
r = 0.06
batch_size = 16
num_epochs = 20000
learning_rate = 0.1
# np.exp(r*t)*st*c_predict to be true value


# In[7]:

with tf.name_scope("placehoders"):
    m = tf.placeholder(tf.float32, [None,1], name='m')
    t = tf.placeholder(tf.float32, [None,1], name='t')
    c = tf.placeholder(tf.float32, [None,1], name='c')
    st = tf.placeholder(tf.float32, [None,1], name='st')


# In[8]:




# In[8]:

w =  tf.Variable(tf.truncated_normal([num_gates, num_neurons,1]), name="weights")
wm = tf.Variable(tf.truncated_normal([1, num_neurons*num_gates]), name="weights_m")
bm = tf.Variable(tf.truncated_normal([num_neurons, num_gates]), name="bias_m")
wt = tf.Variable(tf.truncated_normal([1, num_neurons*num_gates]), name="weights_t")
bt = tf.Variable(tf.truncated_normal([num_neurons, num_gates]), name="bias_t")
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


# In[9]:

exp_wt_can = tf.matmul(t, tf.exp(wt, name="exp_wt_can"))
exp_wt = tf.reshape(exp_wt_can,[-1,num_neurons,num_gates], name="exp_wt")
sigma_t = tf.nn.sigmoid(tf.transpose(exp_wt + bt, [2,0,1]),name="sigma_t")


# In[10]:

exp_wm_can = tf.matmul(m, tf.exp(wm, name="exp_wm_can"))
exp_wm = tf.reshape(exp_wm_can, [-1, num_neurons, num_gates], name="exp_wm")
sigma_m = tf.nn.softplus(tf.transpose(-1.0*exp_wm+bm, [2,0,1]), name="sigma_m")
c_predict = tf.matmul(tf.multiply(sigma_m,sigma_t, name="mt_multi"),tf.exp(w,name="exp_w"), name="c_predict")
single_out = tf.transpose(tf.squeeze(c_predict), name="single_out") #[gates, bs, 1] -> [gates,bs] -> [bs,gates]


# In[11]:

wg1 = tf.Variable(tf.truncated_normal([2, num_gate_neurons]), name="wg1")
bg1 = tf.Variable(tf.truncated_normal([num_gate_neurons]), name="bg1")
wgo = tf.Variable(tf.truncated_normal([num_gate_neurons, num_gates]), name="wgo")
bgo = tf.Variable(tf.truncated_normal([num_gates]), name="bgo")


# In[12]:

inputs = tf.concat([m,t], 1)
h = tf.nn.sigmoid(tf.matmul(inputs,wg1)+bg1, name="h")
gates = tf.nn.softmax(tf.matmul(h, wgo)+bgo, name="softmax_gates")


# In[13]:

c_multi_predict = tf.reduce_sum(tf.multiply(gates,single_out, name="mixture-inner"), 1, name="c_multi_predict")


# In[14]:

#c_predict = tf.matmul(tf.multiply(sigma_m,sigma_t, name="mt_multi"),tf.exp(w,name="exp_w"), name="c_predict")
c_predict_real = tf.exp(-1.0*r*t)*st*c_multi_predict
loss = tf.losses.mean_squared_error(c_predict_real, c)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name="optimizer",global_step=global_step)


# In[15]:

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("histogram_loss", loss)
    summary_op = tf.summary.merge_all()


# In[16]:

import os
init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("log/tf_multi_{}".format(learning_rate), sess.graph)
    saver = tf.train.Saver(max_to_keep=30)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints_multi/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)
    start = global_step.eval()
    for step in range(start, num_epochs):
        _, loss_value, summary = sess.run([optimizer, loss, summary_op], {m:m_values, t:t_values, st:s_values, c:c_values})
        writer.add_summary(summary, global_step=step)
        if(step % 10 == 0):
            print('Loss at step {}: {}'.format(step, loss_value))
            loss_test = sess.run([loss], {m:m_test_values, t:t_test_values, st:s_test_values, c:c_test_values})
            print('Loss at test {}: {}'.format(step, loss_test))    
            saver.save(sess, 'checkpoints_multi/model', step)
            print('Saved model at step {} with test loss: {}'.format(step, loss_test))
    print('Finished training.')
    c_predict_values = sess.run([c_predict_real], {m:m_test_values, t:t_test_values, st:s_test_values, c:c_test_values})
    writer.close()


# In[15]:

c_predict_values


# In[16]:

c_predict_values[0].shape
c_test_values.shape


# In[15]:

c_test_values.shape
result = pd.DataFrame(c_test_values, columns=["c_values"])
result["c_predict_values"] = c_predict_values[0]


# In[16]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[17]:

result.describe()
(result["c_values"] - result["c_predict_values"]).hist(bins=20)
plt.show()


# In[76]:




# In[12]:

mt_combine = tf.concat([m,t], axis=1)
wg = tf.Variable(tf.truncated_normal([2,num_gate_neurons]), name="weights_gate")
bg = tf.Variable(tf.truncated_normal([num_gate_neurons]), name = "bias_gate")
hidden= tf.nn.sigmoid(tf.matmul(mt_combine, wg) +bg)
wgo = tf.Variable(tf.truncated_normal([num_gate_neurons, num_gates]), name="wg_output")
bgo = tf.Variable(tf.truncated_normal([num_gates]), name = "bg_output")
logits = tf.matmul(hidden, wgo) + bgo
gates = tf.nn.softmax(logits, name="gates")


# 
