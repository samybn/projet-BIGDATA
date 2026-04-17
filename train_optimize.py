import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
from mamba_minimal import MambaConfig, MambaModel, MambaClassifier
import re
from tqdm import tqdm
import joblib

# We'll rely on the vectorizer's built-in English stop words.
stop_words = set()

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join([w for w in text.split() if w and w not in stop_words])
    return text

def get_text_column(df):
    # Prefer common names, else pick the first object dtype column
    preferred = ['text', 'content', 'article', 'title', 'body']
    for p in preferred:
        if p in df.columns:
            return p
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        return obj_cols[0]
    raise ValueError('No text-like column found in dataframe')


def prepare_data(test_size=0.2, max_features=2000):
    print("Chargement des données...")
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)

    print("Nettoyage du texte...")
    tqdm.pandas(desc="Nettoyage")
    text_col = get_text_column(df)
    df[text_col] = df[text_col].fillna('').astype(str)
    df['cleaned_text'] = df[text_col].progress_apply(clean_text)

    print("Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text']).astype(np.float32).toarray()
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Ajout d'une dimension de séquence (seq_len = 1)
    X_train = X_train[:, np.newaxis, :]
    X_test  = X_test[:, np.newaxis, :]
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, lr=1e-3):
    device = torch.device('cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    # Move tensors to device
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Cleanup
        del outputs, loss
        gc.collect()

        # Évaluation sur l'ensemble de validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_acc = accuracy_score(y_val_t.cpu().numpy(), val_outputs.argmax(dim=1).cpu().numpy())
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(dataloader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1. Préparation des données (TF-IDF avec 2000 caractéristiques)
    # ------------------------------------------------------------
    MAX_FEATURES = 1000 # Réduit pour économiser la mémoire
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(max_features=MAX_FEATURES)

    # Sous-ensemble pour une optimisation plus rapide (Rduit pour la mmoire)
    SUBSET_SIZE = 500
    subset_size = min(SUBSET_SIZE, len(X_train))
    X_train_sub_all = X_train[:subset_size]
    y_train_sub_all = y_train[:subset_size]
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_sub_all, y_train_sub_all, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------
    # 2. Recherche aléatoire des hyperparamètres
    # ------------------------------------------------------------
    param_distributions = {
        'n_layer': randint(1, 5),           # 1, 2, 3 ou 4 couches
        'lr': uniform(1e-5, 1e-2 - 1e-5),   # entre 1e-5 et 1e-2
        'batch_size': randint(16, 129)      # entre 16 et 128
    }

    n_iter = 20               # nombre de combinaisons testées
    best_acc = 0.0
    best_params = None

    print("=== RECHERCHE ALÉATOIRE DES HYPERPARAMÈTRES ===")
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)

    for i, params in enumerate(sampler):
        print(f"\n--- Essai {i+1}/{n_iter} ---")
        print(f"Params: n_layer={params['n_layer']}, lr={params['lr']:.5f}, batch_size={params['batch_size']}")

        # Création du modèle (d_model fixé à la dimension des données)
        model = MambaClassifier(d_model=MAX_FEATURES, n_layer=params['n_layer'], num_classes=2)

        # Entraînement rapide (3 époques seulement pour l'optimisation)
        model = train_model(model, X_train_sub, y_train_sub, X_val_sub, y_val_sub,
                            epochs=3, batch_size=params['batch_size'], lr=params['lr'])

        # Évaluation sur la validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_sub, dtype=torch.float32)
            outputs = model(X_val_t)
            acc = accuracy_score(y_val_sub, outputs.argmax(dim=1).numpy())
        print(f"Précision sur validation : {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print("\n=== MEILLEURS HYPERPARAMÈTRES TROUVÉS ===")
    if best_params is None:
        print("Aucun meilleur param trouvé: utilisation de valeurs par défaut")
        best_params = {'n_layer': 1, 'lr': 1e-3, 'batch_size': 32}
    print(f"n_layer = {best_params['n_layer']}")
    print(f"lr = {best_params['lr']:.5f}")
    print(f"batch_size = {best_params['batch_size']}")
    print(f"Précision validation = {best_acc:.4f}")

    # ------------------------------------------------------------
    # 3. Entraînement final avec les meilleurs hyperparamètres
    # ------------------------------------------------------------
    print("\n=== ENTRAÎNEMENT FINAL SUR TOUT L'ENSEMBLE D'ENTRAÎNEMENT ===")
    final_model = MambaClassifier(d_model=MAX_FEATURES,
                                  n_layer=best_params['n_layer'],
                                  num_classes=2)
    final_model = train_model(final_model, X_train, y_train, X_test, y_test,
                              epochs=10,
                              batch_size=best_params['batch_size'],
                              lr=best_params['lr'])

    # Évaluation sur l'ensemble de test
    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        outputs = final_model(X_test_t)
        test_acc = accuracy_score(y_test, outputs.argmax(dim=1).numpy())
    print(f"\nPrécision finale sur l'ensemble de test : {test_acc:.4f}")

    # ------------------------------------------------------------
    # 4. Sauvegarde du modèle et du vectoriseur
    # ------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'd_model': MAX_FEATURES,
        'n_layer': best_params['n_layer'],
        'vectorizer': vectorizer
    }, 'models/mamba_fake_news.pth')
    # Save vectorizer separately for easy reuse
    try:
        joblib.dump(vectorizer, 'models/vectorizer.joblib')
    except Exception:
        pass
    print("Modèle sauvegardé dans 'models/mamba_fake_news.pth'")