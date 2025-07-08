import pickle

def load_global_model(path="fl_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
