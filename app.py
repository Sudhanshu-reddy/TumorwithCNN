from flask import Flask, render_template, request, redirect, url_for
import os, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load Model
json_file = open('Model/model.json', 'r')
model = model_from_json(json_file.read())
json_file.close()
model.load_weights("Model/model_weights.h5")

disease = ['No Tumor', 'Tumor']

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 255, 255), 2)
    return result

def tumorSegmentation(filename):
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1)
    img = (img - 127.0) / 127.0
    # Assuming your model predicts binary mask (for example)
    preds = model.predict(img)[0]
    orig = cv2.imread(filename, 0)
    orig = cv2.resize(orig, (300, 300))
    cv2.imwrite("test1.png", orig)
    segmented_image = cv2.resize(preds, (300, 300))
    cv2.imwrite("myimg.png", segmented_image * 255)
    edge_image = edgeDetection()
    return segmented_image * 255, edge_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    with open('Model/history.pckl', 'rb') as f:
        hist = pickle.load(f)
    acc = hist['accuracy']
    los = hist['loss']
    val_acc = hist['val_accuracy']
    val_los = hist['val_loss']

    plt.figure(figsize=(10,5))
    plt.plot(acc, label="Train Accuracy")
    plt.plot(los, label="Train Loss")
    plt.plot(val_acc, label="Val Accuracy")
    plt.plot(val_los, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/line_chart.png")
    plt.close()

    last_values = [acc[-1], los[-1], val_acc[-1], val_los[-1]]
    labels = ['Accuracy', 'Loss', 'Val Accuracy', 'Val Loss']

    plt.figure(figsize=(6,4))
    plt.bar(labels, last_values, color='skyblue')
    plt.ylim(0, 1)
    plt.title("Last Epoch Results")
    plt.tight_layout()
    plt.savefig("static/bar_chart.png")
    plt.close()

    return render_template('train.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, 128, 128, 1).astype('float32') / 255.0
        prediction = model.predict(img)
        cls = np.argmax(prediction)

        result_img_path = os.path.join(app.config['OUTPUT_FOLDER'], 'result.png')
        segmented_img_path = ""
        edge_img_path = ""

        if cls == 1:
            segmented, edge = tumorSegmentation(filepath)
            cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'segmented.png'), segmented)
            cv2.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'edge.png'), edge)
            segmented_img_path = 'output/segmented.png'
            edge_img_path = 'output/edge.png'

        result_img = cv2.imread(filepath)
        cv2.putText(result_img, f'Prediction: {disease[cls]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imwrite(result_img_path, result_img)

        return render_template('test.html', 
                               result='Prediction: ' + disease[cls],
                               input_img='uploads/' + filename,
                               result_img=result_img_path,
                               segmented_img=segmented_img_path,
                               edge_img=edge_img_path)
    return render_template('test.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
