# Offline Pashto Handwriting Detection (PHTI)

A professional machine learning pipeline for offline Pashto handwriting recognition. It leverages YOLOv8 for precise text-line segmentation and a Convolutional Recurrent Neural Network (CRNN) with CTC loss for sequence recognition.

## Project Structure

- `app.py`: Flask Web Application providing a user-friendly interface for document uploads and OCR results.
- `train.py`: Main script for training the CRNN model on the dataset.
- `predict.py`: CLI inference script for running recognition on individual images.
- `augment_for_mobile.py`: Generates synthetic mobile-photo degradations (shadows, perspective warp, noise) to improve model robustness.
- `finetune_robust.py`: Fine-tunes the base model against mobile-photo noise.
- `src/`: Core logic package:
  - `dataset.py`: PyTorch Dataset implementation and data augmentations.
  - `model.py`: CRNN architecture (CNN + BiLSTM).
  - `pipeline.py`: Orchestrates the two-stage inference (Segment -> Recognize).
  - `segmenter.py`: YOLOv8-based line detection with algorithmic fallbacks.
- `models/`: Storage for trained weights (`.pth`, `.pt`).
- `templates/`: Contains Jinja2 HTML layouts/pages for the web interface.
- `static/`: Contains static frontend assets (CSS, uploaded images, etc.).

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Web Interface
Start the Flask application:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your web browser.

### 2. Model Training
To train the base CRNN:
```bash
python train.py
```

### 3. CLI Inference
To test on a single line crop:
```bash
python predict.py
```

---

## Dataset Overview (PHTI)

The PHTI is a real dataset developed for the research community, considering the recognition systems' generalization. The overall process includes; the data collection process, image acquisition process, text-line segmentation process, annotation process and detailed statistics of the PHTI dataset. The PHTI dataset is developed to cover many Pashto language genres such as prose, poetry, short story, history, sports, BBC Pashto, and religion. In this context, data were collected from a diverse background of learners having different levels of educational background, including University, College, School, and Deeni Madaras. 

Each collected handwritten sample is scanned with 600 DPI using an HP Scanjet scanner. The acquired image is named `PHTI-XX-Y-ZZZZ.jpg` where `XX` annotates the source name, `Y` represents gender, and `ZZZZ` describes the page number. The PHTI images are segmented into 36,082 text-line images.

### Statistics:
- **Total Writers:** 400 (200 Female, 200 Male)
- **Total Pages:** 3,970
- **Total Text-Line Images:** 36,082
- **Unique Words:** 33,330

## Applications
1. Natural Language Processing
2. Gender and Age Classification
3. Skew and Line Segmentation
4. OCR Applications

The dataset is licensed under the GNU General Public License v3.0.
