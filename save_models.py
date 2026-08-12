"""
save_models.py
Treina os 4 modelos (com os melhores hiperparâmetros já encontrados pelo benchmark)
nos recortes de treino e salva em models/ para a DEMONSTRAÇÃO.
Rápido: usa os params ótimos direto, sem GridSearch.
"""
import os, pickle
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

CLASSES = ['empty', 'occupied']
os.makedirs('models', exist_ok=True)


def load(split, size):
    X, y = [], []
    for lab, c in enumerate(CLASSES):
        d = os.path.join('crops', split, c)
        for f in os.listdir(d):
            im = cv2.imread(os.path.join(d, f))
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (size, size)), cv2.COLOR_BGR2RGB)
            X.append(im)
            y.append(lab)
    return np.asarray(X, dtype=np.float32), np.asarray(y)


# ---- clássicos (15x15 achatado, normalizado) ----
print("Treinando modelos clássicos (15x15)...")
X15, y = load('train', 15)
Xf = (X15 / 255.0).reshape(len(X15), -1)
classical = {
    'svm': SVC(C=10, gamma=0.01),
    'knn': KNeighborsClassifier(n_neighbors=7, metric='manhattan', n_jobs=-1),
    'rf':  RandomForestClassifier(n_estimators=100, criterion='entropy',
                                  random_state=42, n_jobs=-1),
}
for name, clf in classical.items():
    clf.fit(Xf, y)
    pickle.dump(clf, open(f'models/{name}.pkl', 'wb'))
    print(f"  models/{name}.pkl salvo")

# ---- CNN (32x32) ----
print("Treinando CNN (32x32)...")
import tensorflow as tf
from tensorflow.keras import layers, models
tf.keras.utils.set_random_seed(42)
X32, y32 = load('train', 32)
X32 = X32 / 255.0
cnn = models.Sequential([
    layers.Input((32, 32, 3)),
    layers.Conv2D(32, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation='relu'), layers.BatchNormalization(),
    layers.Dropout(0.5), layers.Dense(1, activation='sigmoid'),
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X32, y32, epochs=12, batch_size=64, verbose=2)
cnn.save('models/cnn.h5')
print("  models/cnn.h5 salvo")
print("\nModelos prontos em models/")
