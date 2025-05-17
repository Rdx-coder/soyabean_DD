from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from predict import predict_disease, load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model once at startup
model = load_model('model/soybean_model.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = predict_disease(filepath, model)
            return render_template('result.html', 
                                   label=result["Disease"], 
                                   solution=result["Recommended Solution"], 
                                   video_url=result["Video"])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
