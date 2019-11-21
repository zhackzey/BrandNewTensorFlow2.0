import tensorflow as tf
import timeit
n = 10**7
with tf.device("/cpu:0"):
    cpu_a = tf.random.normal([1,n])
    cpu_b = tf.random.normal([n,1])
    print(cpu_a.device, cpu_b.device)

def  cpu_run():
    with tf.device("/cpu:0"):
        c = tf.matmul(cpu_a, cpu_b)
    return c


cpu_time = timeit.timeit(cpu_run, number = 10)
print('warmup:', cpu_time)