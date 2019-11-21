import tensorflow as tf
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape: #构建梯度环境
    tape.watch([w]) #将w添加到梯度跟踪列表
    y = a * w ** 2 + b * w + c

[dy_dw] = tape.gradient(y, [w])
print(dy_dw)


     