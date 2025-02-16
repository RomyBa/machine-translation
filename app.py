from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
MAX_LENGTH = 14
MODEL_PATH = "model/machine_translation.keras"
TOKENIZER_EN_PATH = "model/tokenizer_en.pkl"
TOKENIZER_FR_PATH = "model/tokenizer_fr.pkl"

# Load resources at startup
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_EN_PATH, 'rb') as f:
    tokenizer_en = pickle.load(f)
with open(TOKENIZER_FR_PATH, 'rb') as f:
    tokenizer_fr = pickle.load(f)

def translate_sentence(text):
    sequence = tokenizer_en.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    
    decoder_input = np.zeros((1, MAX_LENGTH))
    # translated = []
    
    for i in range(MAX_LENGTH):
        pred = model.predict([padded, decoder_input], verbose=0)
        predicted_token = np.argmax(pred[0, i])
        # if idx == tokenizer_fr.word_index.get('<end>', 0): break
        # translated.append(tokenizer_fr.index_word.get(idx, ''))
        # decoder_input[0, i+1] = idx
        decoder_input[0, i] = predicted_token
    translated_sentence = ' '.join([tokenizer_fr.index_word[token] for token in decoder_input[0] if token != 0])
    
    return translated_sentence #' '.join(translated)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def handle_translation():
    text = request.form.get('text', '').strip()

    if not text:
        logger.warning("No text received for translation.")
        return jsonify({'error': 'No input text provided'}), 400

    logger.info(f"Received text: {text}")  # Log received text

    try:
        translation = translate_sentence(text)
        logger.info(f"Translation result: {translation}")
        return jsonify({'translation': translation})
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

