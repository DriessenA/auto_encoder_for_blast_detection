from pytorch_experiments.models.factory import ModelConstructorRegistry
from project_code import model_lib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


model_registry = ModelConstructorRegistry()
model_registry.auto_register(model_lib)
model_registry.register("MLP", MLPClassifier)
model_registry.register("SVM", SVC)
model_registry.register("RF", RandomForestClassifier)
model_registry.register("LM", LogisticRegression)
