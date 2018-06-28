from keras.layers import Input, Dense, Activation, Conv1D
from keras.models import Model, Sequential
from keras.optimizers import Adam
import numpy as np

d = np.loadtxt('human.txt', dtype=str)
A = []
B = []
for x in d:
    if len(x) > 70:
        z = np.array([int(c) for c in x])
        A.append(z[-70:-20])
        B.append(z[-20:])

x = np.array(A[:800])
y = np.array(B[:800])
tx = np.array(A[800:])
ty = np.array(B[800:])

model = Sequential()
l1 = Dense(50, input_shape=(50,))
model.add(l1)
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(20))

model.compile(optimizer=Adam(), loss='hinge')
model.fit(x=x, y=y, batch_size=32, nb_epoch=10)

f = model.predict(tx)
print(f[0])
print(ty[0])
