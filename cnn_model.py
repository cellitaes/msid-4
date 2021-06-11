from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam
import file_reader

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=3,
                          verbose=1)

# load train data from file
X_train, y_train = file_reader.load_mnist('fashion', kind='train')
# load validation data from file
X_val, y_val = file_reader.load_mnist('fashion', kind='t10k')

# rescale pixel data from 0 to 1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

# changing train and validation data's dimension
X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_val = X_val.reshape(X_val.shape[0], *im_shape)

# defining the model - Sequential model
cnn_model = Sequential([
    # filters - number specifies the output dimensions of the layer
    # kernel size - sliding 3x3 whole image. High value = high similarity
    # activation - Applies the rectified linear unit activation function.
    # input shape - image shape
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape, padding='same'),
    # MaxPooling2D - downsampling the output
    # pool_size - instead of 28 we're going to have 14
    MaxPooling2D(pool_size=2, padding='same'),
    # randomly drop out certain connections to the next layer
    Dropout(0.2),

    # flatten all the layers
    Flatten(),
    Dense(32, activation='relu'),
    # output dimension = 10 for the number of outputs we need
    # On output layer we use Softmax function because our output is multidimensional.
    # Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities
    Dense(10, activation='softmax'),
])

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    # learning rate = 0.001
    optimizer=Adam(lr=0.001),
    # maximize the accuracy
    metrics=['accuracy'],
)

# train data: X_train, y_train
# validation data: (X_val, y_val
# batch size = 512
# 'training number': 10
cnn_model.fit(
    X_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStop],
)

score = cnn_model.evaluate(X_val, y_val, verbose=1)
print(f'test loss: {score[0]}')
print(f'test accu: {score[1]}')
