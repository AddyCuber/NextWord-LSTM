NextWord-LSTM: Predicting the Next Word with LSTMs

Introduction

NextWord-LSTM is a deep learning project that uses Long Short-Term Memory (LSTM) networks to predict the next word in a given sequence. This model is trained on Shakespeare’s Hamlet and demonstrates how NLP techniques can generate text with contextual understanding.

Features

✅ Word-Level LSTM Model – Predicts the next word based on input text.
✅ Trained on Hamlet – Uses the full text of Hamlet from the NLTK Gutenberg corpus.
✅ Pre-Trained Model Included – Easily load and test the trained model (model.h5).
✅ Tokenization & Padding – Efficient preprocessing using Keras Tokenizer and padding techniques.

Project Structure
📂 NextWord-LSTM  
 ├── 📄 README.md        # Project documentation  
 ├── 📄 hamlet.txt       # Dataset: Full text of Hamlet  
 ├── 📄 train.ipynb      # Jupyter Notebook for training the model  
 ├── 📄 model.h5         # Trained LSTM model  
 ├── 📄 tokenizer.pkl    # Tokenizer for text preprocessing  

Installation & Usage

1. Clone the Repository
   git clone https://github.com/AddyCuber/NextWord-LSTM.git
   cd NextWord-LSTM

2. Install Dependencies
   pip install tensorflow numpy pandas nltk scikit-learn

3. Load the Model & Tokenizer
   import pickle
   import tensorflow as tf
   from tensorflow.keras.preprocessing.sequence import pad_sequences

   # Load trained model
   model = tf.keras.models.load_model("model.h5")

   # Load tokenizer
   with open("tokenizer.pkl", "rb") as f:
     tokenizer = pickle.load(f)

4. Predict the Next Word
    def predict_next_word(model, tokenizer, text, max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
        
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index[0]:
                return word
        return None
    
    input_text = "To be or not to be"
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len=20)
    print(f"Predicted next word: {next_word}")
