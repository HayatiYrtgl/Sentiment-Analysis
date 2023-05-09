from artificial.Natural_lang_process.nlp_2_sarcasm import *
from keras.models import Sequential
from keras.layers import Embedding, GlobalAvgPool1D, Dense
import matplotlib.pyplot as plt
import numpy as np


# model constructor

def sarcastic_model():
    # model

    model = Sequential()

    # adding model required layers (vocab lenght 10k, output_dim 128 , input lenght 152)

    model.add(Embedding(input_dim=40000, output_dim=128, input_length=200))

    model.add(GlobalAvgPool1D())

    model.add(Dense(24, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])

    return model


model_ = sarcastic_model()


def start_training(sarcasm_model):
    # start_training function to train data sets

    history = sarcasm_model.fit(training_padded, np.array(train_labels), epochs=30, validation_data=(testing_padded,
                                                                                                     np.array(test_labels)),
                                verbose=1)

    # saving model

    sarcasm_model.save("model_weights/Sentiment_model.h5")

    return history

# train and return history values to plotting


trained_history_value = start_training(model_)


def plotting(history_values):

    # visualize the acc loss

    # main acc loss values

    acc = history_values.history["acc"]

    loss = history_values.history["loss"]

    # validation values

    val_acc = history_values.history["val_acc"]

    val_loss = history_values.history["val_loss"]

    # epoch num for x label

    epoch_num = range(1, 31)

    """PLOT 1"""

    plt.subplot(1, 2, 1)

    # accuracy

    plt.plot(epoch_num, acc, label="Accuracy")

    plt.savefig("plots/accuracy.png")

    # loss

    plt.subplot(1, 2, 2)

    plt.plot(epoch_num, loss, label="Loss")

    plt.savefig("plots/loss.png")

    """PLOT 2"""

    # val_accuracy

    plt.subplot(2, 2, 1)

    plt.plot(epoch_num, val_acc, label="val_accuracy")

    plt.savefig("plots/val_acc")

    # val_loss

    plt.subplot(2, 2, 2)

    plt.plot(epoch_num, val_loss, label="val_loss")

    plt.savefig("plots/val_loss")


# run plotting function with trained hist values

plotting(trained_history_value)


