from keras.models import load_model

def load_my_model():
    model_path = "efficientnetB3_model_15_May_v1.keras"
    model = load_model(model_path)
    return model
