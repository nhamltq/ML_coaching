from abc import ABC, abstractmethod


class BaseModelLinear(ABC):
    def __init__(
        self, model_path, Xtrain: None, ytrain: None, Xtest: None, ytest: None
    ):
        self.model_path = model_path
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        if Xtest is not None and ytest is not None:
            self.Xtest = Xtest
            self.ytest = ytest

    @abstractmethod
    def split_data(self):
        pass

    @abstractmethod
    def cleaning_data(self):
        pass

    @abstractmethod
    def training(self):
        pass

    @abstractmethod
    def load_model(self):
        pass