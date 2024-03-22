from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('salary.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
le_job = data["le_job"]

@app.route('/')
def homepage():
    return render_template('salary.html')

@app.route('/estimate',methods = ['POST'])
def estimator():
    d1 = request.form['education_level']
    d2 = request.form['job_title']
    d3 = request.form['year_of_experience']
    
    if d1 == "Bachelor's":
        d1 = 2
    if d1 == "Master's":
        d1 = 1
    if d1 == 'PhD':
        d1 =0
        
    inp = np.array([[d1,d2,d3]])
    inp[:,1] = le_job.transform(inp[:,1])
    inp = inp.astype(int)
    
    prediction = model.predict(inp)
    prediction = np.round(prediction,2)
    
    text = f'The esimated salary is ${prediction}'
    
    return render_template('salary.html',pred = text)

if __name__ == "__main__":
    app.run(debug=True)
