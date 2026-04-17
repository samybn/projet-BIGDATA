import os
import re
import torch
import nltk
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from nltk.corpus import stopwords
from mamba_minimal import MambaClassifier

app = Flask(__name__, static_folder='ui')
app.config['SECRET_KEY'] = 'detect-info-secret-88'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
CORS(app)

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize Database
with app.app_context():
    db.create_all()

# --- Model Loading Logic ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

MODEL_PATH = 'models/mamba_fake_news.pth'
model = None
vectorizer = None

def load_model_if_needed():
    global model, vectorizer
    if model is None or vectorizer is None:
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            d_model = checkpoint['d_model']
            n_layer = checkpoint['n_layer']
            vectorizer = checkpoint['vectorizer']
            model = MambaClassifier(d_model=d_model, n_layer=n_layer, num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

# --- Routes ---

@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

# Auth API
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Données incomplètes'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Cet utilisateur existe déjà'}), 400
    
    hashed_pw = generate_password_hash(data['password'])
    new_user = User(username=data['username'], password_hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    
    login_user(new_user)
    return jsonify({'success': True, 'username': new_user.username})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    
    if user and check_password_hash(user.password_hash, data.get('password')):
        login_user(user)
        return jsonify({'success': True, 'username': user.username})
    
    return jsonify({'error': 'Identifiants invalides'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})

@app.route('/api/user', methods=['GET'])
def get_user():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

# Prediction API
@app.route('/api/predict', methods=['POST'])
def predict():
    if not load_model_if_needed():
        return jsonify({'error': 'Modèle non trouvé. Veuillez d\'abord entraîner le modèle.'}), 404
    
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'Aucun texte fourni'}), 400
    
    text = data['text']
    cleaned = clean_text(text)
    
    try:
        X = vectorizer.transform([cleaned]).toarray()
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            outputs = model(X_t)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
        
        # Labels updated according to user request
        label = "Real News" if pred_class == 1 else "Fake News"
        confidence = float(probs[0, pred_class].item())
        
        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'is_true': bool(pred_class == 1)
        })
    except Exception as e:
        return jsonify({'error': f"Erreur de prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
