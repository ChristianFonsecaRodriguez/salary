# Usando dataset pycaret
from pycaret.classification import *
import gradio as gr

model = load_model('salary')

def predict_charges(age, fnlwgt, education_num, 
                    capital_gain, capital_loss, hours_per_week):
    data_dict = {'age':[age],'fnlwgt':[fnlwgt], 'education-num':[education_num], 
                'capital-gain':[capital_gain], 'capital-loss':capital_loss, 
                'hours-per-week':hours_per_week}
    df = pd.DataFrame(data_dict)
    pred = predict_model(model, df)['Score'][0]
    return pred

def main():
    age = gr.inputs.Slider(minimum=18, maximum=90, default=30, label='age')
    fnlwgt = gr.inputs.Number(label='fnlwgt')
    education_num = gr.inputs.Slider(minimum=1, maximum=16, default=10, label='education_num')
    capital_gain = gr.inputs.Number(label='capital_gain')
    capital_loss = gr.inputs.Number(label='capital_loss')
    hours_per_week = gr.inputs.Slider(minimum=1, maximum=99, default=40, label='hours_per_week')

    gr.Interface(predict_charges, 
                 inputs=[age, fnlwgt, education_num, 
                         capital_gain, capital_loss, hours_per_week],
                 outputs='label').launch()

if __name__ == '__main__':
    main()


