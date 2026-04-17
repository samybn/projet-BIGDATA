import torch
import re
import nltk
from nltk.corpus import stopwords
from mamba_minimal import MambaClassifier

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_model(model_path='models/mamba_fake_news.pth'):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    d_model = checkpoint['d_model']
    n_layer = checkpoint['n_layer']
    vectorizer = checkpoint['vectorizer']
    model = MambaClassifier(d_model=d_model, n_layer=n_layer, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vectorizer

def predict_text(model, vectorizer, text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned]).toarray()
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (1, 1, features)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()
    return pred_class, confidence

if __name__ == "__main__":
    print("Chargement du modèle...")
    model, vectorizer = load_model()
    print("Modèle chargé. Tapez 'quit' pour quitter.\n")
    while True:
        text = input("Entrez une phrase à analyser : ")
        if text.lower() == 'quit':
            break
        pred, conf = predict_text(model, vectorizer, text)
        label = "FAKE NEWS" if pred == 0 else "INFORMATION VÉRIDIQUE"
        print(f"Résultat : {label} (confiance: {conf:.2%})\n")