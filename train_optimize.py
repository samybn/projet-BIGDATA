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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
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

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, lr=1e-3, use_scheduler=False):
    device = torch.device('cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

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

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        diverged = False
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                print(f"!!! ATTENTION: NaN détecté à l'époque {epoch+1}. Arrêt de la séquence.")
                diverged = True
                break
                
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if diverged:
            break

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_acc = accuracy_score(y_val_t.cpu().numpy(), val_outputs.argmax(dim=1).cpu().numpy())
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Val Acc: {val_acc:.4f}")
        
        if scheduler:
            scheduler.step(val_acc)
        
        best_val_acc = max(best_val_acc, val_acc)

    return model, best_val_acc

class GeneticOptimizer:
    def __init__(self, d_model, X_train, y_train, X_val, y_val, pop_size=6, generations=3):
        self.d_model = d_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pop_size = pop_size
        self.generations = generations
        
        # Plages de paramètres plus sûres
        self.bounds = {
            'n_layer': (1, 5),
            'lr': (1e-5, 5e-3), # Maximum 0.005 pour éviter l'instabilité
            'batch_size': (16, 129)
        }
        
    def create_individual(self):
        return {
            'n_layer': random.randint(*self.bounds['n_layer']),
            'lr': random.uniform(*self.bounds['lr']),
            'batch_size': random.randint(*self.bounds['batch_size'])
        }
    
    def evaluate_fitness(self, ind):
        print(f"   Évaluation: n_layer={ind['n_layer']}, lr={ind['lr']:.5f}, batch_size={ind['batch_size']}")
        model = MambaClassifier(d_model=self.d_model, n_layer=ind['n_layer'], num_classes=2)
        # Entraînement court pour l'évaluation génétique
        model, acc = train_model(model, self.X_train, self.y_train, self.X_val, self.y_val, 
                               epochs=2, batch_size=ind['batch_size'], lr=ind['lr'])
        
        # Pénalité en cas d'effondrement ou de NaNs
        if acc < 0.6 or np.isnan(acc):
            print("   -> ATTENTION: Modèle instable, fitness pénalisée.")
            acc = 0.1

        # Nettoyage mémoire
        del model
        gc.collect()
        return acc

    def crossover(self, p1, p2):
        child = {}
        for key in p1.keys():
            child[key] = p1[key] if random.random() > 0.5 else p2[key]
        return child

    def mutate(self, ind):
        if random.random() < 0.2: # 20% mutation chance
            key = random.choice(list(self.bounds.keys()))
            if key == 'lr':
                ind[key] = random.uniform(*self.bounds[key])
            else:
                ind[key] = random.randint(*self.bounds[key])
        return ind

    def run(self):
        print("\n=== INITIALISATION DE LA POPULATION (GÉNÉRATION 0) ===")
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        best_overall_params = None
        best_overall_fitness = -1.0
        
        for gen in range(self.generations):
            print(f"\n--- GÉNÉRATION {gen + 1} / {self.generations} ---")
            fitness_scores = []
            
            for i, ind in enumerate(population):
                print(f"Individu {i+1}/{self.pop_size}")
                fit = self.evaluate_fitness(ind)
                fitness_scores.append((ind, fit))
                
                if fit > best_overall_fitness:
                    best_overall_fitness = fit
                    best_overall_params = ind.copy()
            
            # Tri par fitness (décroissant)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"Meilleure précision génération {gen+1}: {fitness_scores[0][1]:.4f}")
            
            if gen < self.generations - 1:
                # Évolution: Elitisme (garder les 2 meilleurs) + Crossover/Mutation
                new_population = [fitness_scores[0][0], fitness_scores[1][0]]
                
                while len(new_population) < self.pop_size:
                    # Sélection par tournoi (parent 1)
                    p1 = random.choice(fitness_scores[:3])[0]
                    # Sélection par tournoi (parent 2)
                    p2 = random.choice(fitness_scores[:3])[0]
                    
                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    new_population.append(child)
                
                population = new_population
                
        return best_overall_params, best_overall_fitness

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
    # 2. Optimisation Métaheuristique (Algorithme Génétique)
    # ------------------------------------------------------------
    print("=== OPTIMISATION MÉTAHEURISTIQUE (ALGORITHME GÉNÉTIQUE) ===")
    
    optimizer = GeneticOptimizer(
        d_model=MAX_FEATURES,
        X_train=X_train_sub,
        y_train=y_train_sub,
        X_val=X_val_sub,
        y_val=y_val_sub,
        pop_size=6,
        generations=3
    )
    
    best_params, best_acc = optimizer.run()

    print("\n=== MEILLEURS HYPERPARAMÈTRES TROUVÉS PAR GA ===")
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
    final_model, final_acc = train_model(final_model, X_train, y_train, X_test, y_test,
                                        epochs=12, # Augmenté légèrement car learning rate plus bas
                                        batch_size=best_params['batch_size'],
                                        lr=best_params['lr'],
                                        use_scheduler=True)

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