import abc
class Classifier_interface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and callable(subclass.train) 
            #and hasattr(subclass, 'predict') and callable(subclass.predict) 
            or NotImplemented)

    @abc.abstractmethod
    def train(self, X_train, Y_train, X_valid, Y_valid, batch_size, epochs):
        """Load in the data set"""
        raise NotImplementedError

    #@abc.abstractmethod
    #def predict(self, X):
        """Load in the data set"""
        #raise NotImplementedError