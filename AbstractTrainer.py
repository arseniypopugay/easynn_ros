from abc import ABCMeta, abstractmethod

class Trainer(object):

    __metaclass__=ABCMeta

    def __init__(self, script_path, data, epochs, output_dir):
        self.script_path = script_path
        self.data = data
        self.epochs = epochs
        self.output_dir = output_dir

    @abstractmethod
    def train(self):
        pass
