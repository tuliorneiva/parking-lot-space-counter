"""
benchmark.py
Compara paradigmas de Machine Learning na tarefa de classificar vagas de
estacionamento (vazia x ocupada) usando os recortes gerados por extract_crops.py.

Modelos clássicos (SVM, KNN, Random Forest): imagem RGB reduzida a 15x15 e achatada
(675 features), exatamente como o SVM original do projeto.
CNN: imagem RGB 32x32, arquitetura convolucional treinada do zero.

Todos treinam no MESMO conjunto de treino e são avaliados no MESMO conjunto de teste,
garantindo uma comparação justa. Para diagnosticar overfitting/underfitting, cada
modelo também é avaliado no próprio treino e na validação, e reportamos o "gap"
(treino - teste). Métricas: acurácia, precisão, recall, F1, matriz de confusão e
tempos de treino/inferência.

Uso:
    python benchmark.py --crops crops --out benchmark_results
    python benchmark.py --crops crops --out benchmark_results --cnn-epochs 15
"""
import os
import time
import json
import argparse
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['empty', 'occupied']  # rótulo 0 = empty, 1 = occupied

# limite do gap (acurácia treino - teste) a partir do qual sinalizamos overfitting
OVERFIT_GAP = 0.05
# acurácia de treino abaixo da qual o modelo provavelmente está com underfitting
UNDERFIT_ACC = 0.90



# Carregamento dos dados
def load_split(crops_dir, split, size):
    """Carrega recortes de um split e devolve X (imagens size x size x3) e y."""
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(crops_dir, split, cls)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            img = cv2.imread(os.path.join(d, fname))
            if img is None:
                continue
            if img.shape[:2] != (size, size):
                img = cv2.resize(img, (size, size))
            X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR -> RGB
            y.append(label)
    return np.asarray(X), np.asarray(y)


# Métricas
def scores(y_true, y_pred):
    return {
        'acuracia': accuracy_score(y_true, y_pred),
        'precisao': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


def diagnose(acc_train, acc_test):
    """Diagnóstico simples de overfitting/underfitting."""
    gap = acc_train - acc_test
    if acc_train < UNDERFIT_ACC and acc_test < UNDERFIT_ACC:
        return 'underfitting'
    if gap > OVERFIT_GAP:
        return 'overfitting'
    return 'ok'


def build_result(name, y_tr, p_tr, y_va, p_va, y_te, p_te, train_time, infer_time):
    s_tr, s_va, s_te = scores(y_tr, p_tr), scores(y_va, p_va), scores(y_te, p_te)
    gap = s_tr['acuracia'] - s_te['acuracia']
    diag = diagnose(s_tr['acuracia'], s_te['acuracia'])
    r = {
        'modelo': name,
        'treino': s_tr,
        'validacao': s_va,
        'teste': s_te,
        'gap_treino_teste': gap,
        'diagnostico': diag,
        'tempo_treino_s': train_time,
        'tempo_inferencia_s': infer_time,
        'matriz_confusao': confusion_matrix(y_te, p_te).tolist(),
    }
    print(f"  {name:16s} acc_treino={s_tr['acuracia']:.4f}  "
          f"acc_val={s_va['acuracia']:.4f}  acc_teste={s_te['acuracia']:.4f}  "
          f"gap={gap:+.4f}  -> {diag}")
    return r


# Modelos clássicos
def run_classical(sets, clf_size):
    (Xtr_img, ytr), (Xva_img, yva), (Xte_img, yte) = sets

    def flatten(imgs):
        out = np.empty((len(imgs), clf_size * clf_size * 3), dtype=np.float32)
        for i, im in enumerate(imgs):
            r = cv2.resize(im, (clf_size, clf_size)).astype(np.float32) / 255.0
            out[i] = r.flatten()
        return out

    Xtr, Xva, Xte = flatten(Xtr_img), flatten(Xva_img), flatten(Xte_img)

    models = {
                'SVM': GridSearchCV(
                        SVC(), [{'gamma': [0.01, 0.001],
                                 'C': [1, 10, 100]}], cv=3, n_jobs=-1),
                
                'KNN': GridSearchCV(
                        KNeighborsClassifier(n_jobs=-1),
                        [{'n_neighbors': [7, 11, 13, 17, 19, 23],
                          'weights': ['uniform'],
                          'metric': ['euclidean', 'manhattan', 'minkowski']}],
                        cv=3, n_jobs=-1),
                
                'Random Forest': GridSearchCV(
                                    RandomForestClassifier(random_state=42, n_jobs=-1),
                                    [{'n_estimators': [100, 200, 400],
                                      'max_depth': [None, 10, 20],
                                      "criterion": ["gini", "entropy", "log_loss"]}],
                                    cv=3, n_jobs=-1),
            }

    results = []
    
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_time = time.time() - t0
        t0 = time.time()
        p_te = model.predict(Xte)
        infer_time = time.time() - t0
        p_tr, p_va = model.predict(Xtr), model.predict(Xva)
        
        if isinstance(model, GridSearchCV):
            print(f"  [{name}] melhores params: {model.best_params_}")
            
        results.append(build_result(name, ytr, p_tr, yva, p_va, yte, p_te,
                                     train_time, infer_time))
        
    return results


# CNN
def run_cnn(sets, cnn_size, epochs, batch_size, out_dir):
    from tensorflow.keras import layers, models

    (Xtr_img, ytr), (Xva_img, yva), (Xte_img, yte) = sets

    def prep(imgs):
        out = np.empty((len(imgs), cnn_size, cnn_size, 3), dtype=np.float32)
        for i, im in enumerate(imgs):
            out[i] = cv2.resize(im, (cnn_size, cnn_size)).astype(np.float32) / 255.0
        return out

    Xtr, Xva, Xte = prep(Xtr_img), prep(Xva_img), prep(Xte_img)

    model = models.Sequential([
        layers.Input((cnn_size, cnn_size, 3)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    # validação explícita (mesmo conjunto usado pelos outros modelos)
    history = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size,
                        validation_data=(Xva, yva), verbose=2)
    train_time = time.time() - t0

    t0 = time.time()
    p_te = (model.predict(Xte, verbose=0).ravel() >= 0.5).astype(int)
    infer_time = time.time() - t0
    p_tr = (model.predict(Xtr, verbose=0).ravel() >= 0.5).astype(int)
    p_va = (model.predict(Xva, verbose=0).ravel() >= 0.5).astype(int)

    plot_cnn_history(history.history, out_dir)
    return build_result('CNN', ytr, p_tr, yva, p_va, yte, p_te,
                        train_time, infer_time)


# Gráficos
def plot_cnn_history(hist, out_dir):
    """Curva de aprendizado: treino vs validação. Separação = overfitting."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist['accuracy'], label='treino')
    a1.plot(hist['val_accuracy'], label='validação')
    a1.set_title('CNN — Acurácia por época'); a1.set_xlabel('época')
    a1.set_ylabel('acurácia'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(hist['loss'], label='treino')
    a2.plot(hist['val_loss'], label='validação')
    a2.set_title('CNN — Loss por época'); a2.set_xlabel('época')
    a2.set_ylabel('loss'); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cnn_learning_curve.png'), dpi=120)
    plt.close(fig)


def plot_confusion_matrices(results, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r['matriz_confusao'])
        ax.imshow(cm, cmap='Blues')
        ax.set_title(f"{r['modelo']}\nacc_teste={r['teste']['acuracia']:.3f}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=120)
    plt.close(fig)


def plot_train_vs_test(results, out_dir):
    """Barras de acurácia treino vs teste — visualiza o gap (overfitting)."""
    labels = [r['modelo'] for r in results]
    x = np.arange(len(labels)); w = 0.35
    tr = [r['treino']['acuracia'] for r in results]
    te = [r['teste']['acuracia'] for r in results]
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 5))
    ax.bar(x - w / 2, tr, w, label='treino')
    ax.bar(x + w / 2, te, w, label='teste')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel('acurácia')
    ax.set_title('Treino vs Teste (gap grande = overfitting)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'train_vs_test.png'), dpi=120)
    plt.close(fig)


def plot_metric_bars(results, out_dir):
    metrics = ['acuracia', 'precisao', 'recall', 'f1']
    labels = [r['modelo'] for r in results]
    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 5))
    for i, met in enumerate(metrics):
        vals = [r['teste'][met] for r in results]
        ax.bar(x + (i - 1.5) * w, vals, w, label=met)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score (teste)')
    ax.set_title('Comparação de paradigmas — vaga vazia x ocupada')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'metrics_comparison.png'), dpi=120)
    plt.close(fig)


def write_outputs(results, out_dir):
    cols = ['modelo', 'acc_treino', 'acc_validacao', 'acc_teste', 'gap',
            'diagnostico', 'precisao_teste', 'recall_teste', 'f1_teste',
            'tempo_treino_s', 'tempo_inferencia_s']
    lines = [','.join(cols)]
    for r in results:
        lines.append(','.join([
            r['modelo'],
            f"{r['treino']['acuracia']:.4f}",
            f"{r['validacao']['acuracia']:.4f}",
            f"{r['teste']['acuracia']:.4f}",
            f"{r['gap_treino_teste']:+.4f}",
            r['diagnostico'],
            f"{r['teste']['precisao']:.4f}",
            f"{r['teste']['recall']:.4f}",
            f"{r['teste']['f1']:.4f}",
            f"{r['tempo_treino_s']:.2f}",
            f"{r['tempo_inferencia_s']:.2f}",
        ]))
    with open(os.path.join(out_dir, 'results.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== RESUMO (overfitting/underfitting) ===")
    header = (f"{'Modelo':16s} {'AccTreino':>9s} {'AccVal':>8s} "
              f"{'AccTeste':>9s} {'Gap':>8s} {'Diagnóstico':>13s}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['modelo']:16s} {r['treino']['acuracia']:9.4f} "
              f"{r['validacao']['acuracia']:8.4f} {r['teste']['acuracia']:9.4f} "
              f"{r['gap_treino_teste']:+8.4f} {r['diagnostico']:>13s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', default='crops')
    ap.add_argument('--out', default='benchmark_results')
    ap.add_argument('--load-size', type=int, default=64)
    ap.add_argument('--clf-size', type=int, default=15)
    ap.add_argument('--cnn-size', type=int, default=32)
    ap.add_argument('--cnn-epochs', type=int, default=12)
    ap.add_argument('--cnn-batch', type=int, default=64)
    ap.add_argument('--max-train-per-class', type=int, default=None,
                    help='reduz o treino p/ N por classe (útil p/ demonstrar overfitting)')
    ap.add_argument('--skip-cnn', action='store_true')
    ap.add_argument('--skip-classical', action='store_true',
                    help='roda apenas a CNN (pula SVM/KNN/Random Forest)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Carregando recortes...")
    Xtr, ytr = load_split(args.crops, 'train', args.load_size)
    Xva, yva = load_split(args.crops, 'valid', args.load_size)
    Xte, yte = load_split(args.crops, 'test', args.load_size)

    # embaralha o treino (recortes vêm ordenados por classe)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    # opcional: reduz o treino de forma balanceada (para demonstrar overfitting)
    if args.max_train_per_class:
        keep = []
        for label in (0, 1):
            idx = np.where(ytr == label)[0][:args.max_train_per_class]
            keep.extend(idx)
        keep = rng.permutation(np.array(keep))
        Xtr, ytr = Xtr[keep], ytr[keep]

    print(f"  treino: {len(ytr)} | validação: {len(yva)} | teste: {len(yte)}")
    if len(ytr) == 0 or len(yte) == 0:
        raise SystemExit("Recortes não encontrados. Rode extract_crops.py primeiro.")

    sets = ((Xtr, ytr), (Xva, yva), (Xte, yte))

    results = []
    if not args.skip_classical:
        print("\nModelos clássicos (features 15x15 achatadas):")
        results += run_classical(sets, args.clf_size)

    if not args.skip_cnn:
        print("\nCNN (32x32):")
        results.append(run_cnn(sets, args.cnn_size, args.cnn_epochs,
                               args.cnn_batch, args.out))

    write_outputs(results, args.out)
    plot_confusion_matrices(results, args.out)
    plot_metric_bars(results, args.out)
    plot_train_vs_test(results, args.out)
    print(f"\nSaídas salvas em: {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
