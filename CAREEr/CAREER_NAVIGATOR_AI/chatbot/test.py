import google.generativeai as genai

genai.configure(api_key="AIzaSyAurjcaQWypq4YXFxiIHwzdWzcyL1Y_P2Y")

models = genai.list_models()
for model in models:
    print(model.name)
