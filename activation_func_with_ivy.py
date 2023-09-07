import ivy

#Exponential Activation Function.
def exponential(x):
    return ivy.to_numpy(ivy.exp(x))

#ELU - Exponential Linear Unit
def ELU(x):
    return ivy.to_numpy(ivy.elu(x))

#SELU - Scaled Exponential Linear Unit
def SELU(x):
    return ivy.to_numpy(ivy.selu(x))

#GELU - Gaussian Error Linear Unit
def GELU(x):
    return ivy.to_numpy(ivy.gelu(x))

#MISH - Mish Activation Function
def MISH(x):
    return ivy.to_numpy(ivy.mish(x))

#Sigmoid Activation Function
def sigmoid(x):
    return ivy.to_numpy(ivy.sigmoid(x))

#Swish Activation Function(Variant of Sigmoid)
def swish(x):
    return x*sigmoid(x)

#RELU - Rectified Linear Unit
def RELU(x):
    return ivy.to_numpy(ivy.relu(x))

#Softmax Activation Function
def softmax(x):
    return ivy.to_numpy(ivy.softmax(x))

#Softplus Activation Function
def softplus(x):
    return ivy.to_numpy(ivy.softplus(x))

# Softsign Activation Function
## Recently Added to IVY
def softsign(x):
    return ivy.to_numpy(ivy.softsign(x))

#TanH - Hyperbolic Tangent
def tanh(x):
    return ivy.to_numpy(ivy.tanh(x))

#Linear Activation Function
def linear(x):
    return ivy.to_numpy(x)

#Hard Sigmoid Activation Function
def hard_sigmoid(x):
    return 0 if x < -2.5 else 1 if x > 2.5 else ivy.add(ivy.multiply(0.2,x),0.5)


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
