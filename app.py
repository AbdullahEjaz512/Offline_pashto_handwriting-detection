from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from src.pipeline import FullPagePashtoRecognition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the AI Brain once when the server starts
print("Initializing AI Engine...")
ai_pipeline = FullPagePashtoRecognition(
    yolo_weights="models/best.pt",
    crnn_weights="models/crnn_pashto.pth",
    vocab_path="models/vocab.json"
)

@app.route('/')
def home():
    # Serves the frontend web page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Pass the image to your AI pipeline
            recognized_text = ai_pipeline.process_page(filepath, save_crops=False)
            
            # 2. Return the text to the web browser
            return jsonify({
                'success': True,
                'text': recognized_text
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)