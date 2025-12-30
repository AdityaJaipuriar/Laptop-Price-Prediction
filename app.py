from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open('laptop_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #data from form
    company = request.form.get('company')
    type_name = request.form.get('type_name')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touchscreen = int(request.form.get('touchscreen'))
    ips = int(request.form.get('ips'))
    inches = float(request.form.get('inches'))
    resolution = request.form.get('resolution')
    cpu_speed = float(request.form.get('cpu_speed'))
    cpu_brand = request.form.get('cpu_brand')
    hdd = int(request.form.get('hdd'))
    ssd = int(request.form.get('ssd'))
    gpu_brand = request.form.get('gpu_brand')
    os_group = request.form.get('os_group')

    #Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / inches

    #Create DataFrame for prediction
    query = pd.DataFrame([[company, type_name, ram, weight, touchscreen, ips, ppi, cpu_speed, cpu_brand, hdd, ssd, gpu_brand, os_group]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'CpuSpeed', 'Cpu brand', 'HDD', 'SSD', 'GpuBrand', 'OpSysGroup'])

    #Transform and Predict
    query_transformed = preprocessor.transform(query)
    prediction = model.predict(query_transformed)
    
    #Inverse Log transformation
    result = int(np.expm1(prediction)[0])

    #PASSING request.form back as 'inputs' to keep the values in the UI
    return render_template('index.html', 
                           prediction_text=f'Estimated Price: ₹ {result:,}',
                           inputs=request.form)

if __name__ == "__main__":
    app.run(debug=True)