class Callback:
    """Base class for all Training Callbacks."""
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass

class CallbackList:
    """Manages a list of Callbacks."""
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        
    def append(self, callback):
        self.callbacks.append(callback)
        
    def on_train_begin(self, logs=None):
        for cb in self.callbacks: cb.on_train_begin(logs)
            
    def on_train_end(self, logs=None):
        for cb in self.callbacks: cb.on_train_end(logs)
            
    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks: cb.on_epoch_begin(epoch, logs)
            
    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks: cb.on_epoch_end(epoch, logs)
            
    def on_batch_begin(self, batch, logs=None):
        for cb in self.callbacks: cb.on_batch_begin(batch, logs)
            
    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks: cb.on_batch_end(batch, logs)
