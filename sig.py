# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
# 更多参考 
# git@github.com:dani-garcia/multiview_gpu.git
# 计算X的相似度矩阵
def euclidean_distances(x):
    """Euclidean distances of a tensor
    # Arguments
        x: A tensor or variable.
    # Returns
        A tensor with the euclidean distances of elements of `x`.
    """
    norm = tf.reduce_sum(tf.square(x), 1)

    norm_row = tf.reshape(norm, [-1, 1])
    norm_col = tf.reshape(norm, [1, -1])

    return tf.sqrt(tf.maximum(norm_row + norm_col - 2*tf.matmul(x, x, transpose_b=True), 0.0))
def calc_sig(D,k):
    DD=euclidean_distances(D)
    one = tf.ones_like(DD)
    zero = tf.zeros_like(DD)
    # 计算前2个最大的值，和索引
    top_k_xvals, top_k_indices =tf.nn.top_k(DD,k)

    kth = tf.reduce_min(top_k_xvals,1,keepdims=True)

    # 如果大于kth则为1，否则为0
    TFMat=tf.where(DD<kth,x=zero,y=one)
    # 计算对应距离矩阵中的
    sigMat=tf.multiply(DD, TFMat)

    return sigMat;

if __name__ == "__main__":
    # Declaring matrices
    sess = tf.Session()

    # Create matrix from np array
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))

    sigmat=calc_sig(D,2);


    # DD=euclidean_distances(D)

    # one = tf.ones_like(DD)
    # zero = tf.zeros_like(DD)

    # # 计算前2个最大的值，和索引
    # top_k_xvals, top_k_indices =tf.nn.top_k(DD,2)

    # # 计算每一行的最小值
    # kth = tf.reduce_min(top_k_xvals,1,keepdims=True)
    
    # # 如果大于kth则为1，否则为0
    # bb=tf.where(DD<kth,x=zero,y=one)
    # # 计算对应距离矩阵中的
    # aaa=tf.multiply(DD, bb)

    print(sess.run(sigmat))

