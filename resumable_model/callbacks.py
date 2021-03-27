from tensorflow import keras
from .utils import merge_dicts_with_only_lists_as_values
import pickle
import os

class EpochCounter(keras.callbacks.Callback):
  def __init__(self, counter_path):
    self.counter_path = counter_path
    super(EpochCounter, self).__init__()
  def on_epoch_begin(self, epoch, logs=None):
    # save epoch number to disk
    if os.path.exists(self.counter_path):
        with open(self.counter_path, "wb") as f:
            pickle.dump(epoch, f)
    else:
        f =open(self.counter_path, "wb")
        pickle.dump(epoch, f)
        f.close()

class HistoryLogger(keras.callbacks.Callback):
  def __init__(self, history_path, recovered_history):
    self.recovered_history = recovered_history
    self.history_path = history_path
    super(HistoryLogger, self).__init__()
  def on_epoch_begin(self, epoch, logs=None):
    combined_history = merge_dicts_with_only_lists_as_values([self.recovered_history, self.model.history.history])
    if os.path.exists(self.history_path):
        with open(self.history_path, "wb") as f:
            pickle.dump(combined_history, f)
    else:
        f = open(self.history_path, "wb")
        pickle.dump(combined_history, f)
        f.close()
