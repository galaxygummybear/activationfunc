import math,tensorflow as tf

#Exponential Activation Function.
def exponential(x):
    return math.exp(x)

#ELU - Exponential Linear Unit
def ELU(x):
    alpha = 1.67326324
    return x if x>0 else alpha*(math.exp(x)-1)

#SELU - Scaled Exponential Linear Unit
def SELU(x):
    scale = 1.05070098
    alpha = 1.67326324
    return scale*x if x>0 else scale*alpha*(math.exp(x)-1)

#GELU - Gaussian Error Linear Unit
def GELU(x):
    return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*math.pow(x,3))))

#MISH - Mish Activation Function
def MISH(x):
    return x*math.tanh(softplus(x))

#Sigmoid Activation Function
def sigmoid(x):
    return 1/(1+math.exp(-x))

#Swish Activation Function(Variant of Sigmoid)
def swish(x):
    return x*sigmoid(x)

#RELU - Rectified Linear Unit
def RELU(x):
    return max(x,0)

#Softmax Activation Function
def softmax(x):
    return math.exp(x)/tf.reduce_sum(math.exp(x))

#Softplus Activation Function
def softplus(x):
    return math.log(math.exp(x)+1)

#Softsign Activation Function
def softsign(x):
    return x/(abs(x)+1)

#TanH - Hyperbolic Tangent
def tanh(x):
    math.tanh(x)

#Linear Activation Function
def linear(x):
    return x

#Hard Sigmoid Activation Function
def hard_sigmoid(x):
    return 0 if x < -2.5 else 1 if x > 2.5 else 0.2*x+0.5

print("Exponential Activation Function: {:.3f}".format(exponential(5.0)))
print("ELU Activation Function: {:.3f}".format(ELU(5.0)))
print("SELU Activation Function: {:.3f}".format(SELU(5.0)))
print("GELU Activation Function: {:.3f}".format(GELU(5.0)))
print("MISH Activation Function: {:.3f}".format(MISH(5.0)))
print("Sigmoid Activation Function: {:.3f}".format(sigmoid(5.0)))
print("Swish Activation Function: {:.3f}".format(swish(5.0)))
print("RELU Activation Function: {:.3f}".format(RELU(5.0)))
print("Softmax Activation Function: {:.3f}".format(softmax(5.0)))
print("Softplus Activation Function: {:.3f}".format(softplus(5.0)))
print("Softsign Activation Function: {:.3f}".format(softsign(5.0)))
print("TanH Activation Function: {:.3f}".format(tanh(5.0)))
print("Linear Activation Function: {:.3f}".format(linear(5.0)))
print("Hard Sigmoid Activation Function: {:.3f}".format(hard_sigmoid(5.0)))

