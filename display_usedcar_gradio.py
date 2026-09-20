import gradio as gr
import joblib

model = joblib.load("used_car_random_forest.pkl")

def predict_price_category(car_age, kilometers_driven):

    new_car = [[car_age, kilometers_driven]]

    prediction = model.predict(new_car)

    return prediction[0]


app = gr.Interface(
    fn=predict_price_category,
    inputs=[
        gr.Number(label="Car Age"),
        gr.Number(label="Kilometers Driven")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Used Car Price Prediction",
    description="Random Forest"
)


app.launch(server_name="0.0.0.0", server_port=7860)
