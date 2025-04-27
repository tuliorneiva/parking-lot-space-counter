# train_cnn_model.py
# Script para treinar uma CNN para classificação de vagas de estacionamento

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import argparse

def load_data(input_dir, target_size=(32, 32)):
    """
    Carrega as imagens de vagas vazias e ocupadas.

    Args:
        input_dir: Diretório contendo as pastas 'empty' e 'occupied'
        target_size: Tamanho para redimensionar as imagens

    Returns:
        X: Imagens redimensionadas
        y: Rótulos (0 para vazio, 1 para ocupado)
    """
    categories = ['empty', 'occupied']

    data = []
    labels = []
    filenames = []

    print("Carregando dados...")

    for category_idx, category in enumerate(categories):
        category_dir = os.path.join(input_dir, category)
        if not os.path.exists(category_dir):
            print(f"Diretório não encontrado: {category_dir}")
            continue

        print(f"Carregando imagens da categoria: {category}")
        files = os.listdir(category_dir)
        print(f"  Encontradas {len(files)} imagens")

        for file in files:
            img_path = os.path.join(category_dir, file)
            try:
                img = imread(img_path)

                # Verificar se a imagem tem 3 canais (RGB)
                if len(img.shape) == 2:  # Imagem em escala de cinza
                    img = np.stack((img,) * 3, axis=-1)
                elif img.shape[2] == 4:  # Imagem com canal alpha
                    img = img[:, :, :3]

                # Redimensionar
                img_resized = resize(img, target_size, anti_aliasing=True)

                # Normalizar para [0, 1]
                if img_resized.max() > 1.0:
                    img_resized = img_resized / 255.0

                data.append(img_resized)
                labels.append(category_idx)
                filenames.append(img_path)
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {str(e)}")

    X = np.array(data)
    y = np.array(labels)

    print(f"Dados carregados: {X.shape[0]} imagens")
    print(f"  Vazias: {np.sum(y == 0)}")
    print(f"  Ocupadas: {np.sum(y == 1)}")

    return X, y, filenames

def create_model(input_shape):
    """
    Cria um modelo CNN para classificação de vagas.

    Args:
        input_shape: Forma da entrada (altura, largura, canais)

    Returns:
        Modelo compilado
    """
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Terceira camada convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Camadas densas
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Saída binária
    ])

    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(X, y, batch_size=32, epochs=50, validation_split=0.2, output_dir='model_output'):
    """
    Treina o modelo CNN.

    Args:
        X: Dados de entrada
        y: Rótulos
        batch_size: Tamanho do lote
        epochs: Número de épocas
        validation_split: Fração dos dados para validação
        output_dir: Diretório para salvar o modelo

    Returns:
        Modelo treinado e histórico de treinamento
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Dividir em conjuntos de treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, stratify=y, random_state=42
    )

    print(f"Conjunto de treinamento: {X_train.shape[0]} imagens")
    print(f"Conjunto de validação: {X_val.shape[0]} imagens")

    # Criar gerador de dados com aumento de dados
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Criar o modelo
    input_shape = X_train.shape[1:]
    model = create_model(input_shape)

    # Resumo do modelo
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    # Treinar o modelo
    print("Iniciando treinamento...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Salvar o modelo final
    model.save(os.path.join(output_dir, 'final_model.h5'))

    # Salvar o histórico
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    return model, history

def evaluate_model(model, X, y):
    """
    Avalia o modelo no conjunto de teste.

    Args:
        model: Modelo treinado
        X: Dados de teste
        y: Rótulos de teste

    Returns:
        Métricas de avaliação
    """
    # Avaliar o modelo
    loss, accuracy = model.evaluate(X, y)

    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

    # Fazer previsões
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calcular métricas
    print("Relatório de classificação:")
    print(classification_report(y, y_pred))

    print("Matriz de confusão:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

def plot_training_history(history, output_dir='model_output'):
    """
    Plota o histórico de treinamento.

    Args:
        history: Histórico de treinamento
        output_dir: Diretório para salvar os gráficos
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Plotar acurácia
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='lower right')

    # Plotar perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_examples(X, y, y_pred, filenames, output_dir='model_output', num_examples=10):
    """
    Plota exemplos de classificação.

    Args:
        X: Dados
        y: Rótulos reais
        y_pred: Rótulos previstos
        filenames: Nomes dos arquivos
        output_dir: Diretório para salvar os gráficos
        num_examples: Número de exemplos a plotar
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    # Encontrar exemplos corretos e incorretos
    correct = np.where(y == y_pred)[0]
    incorrect = np.where(y != y_pred)[0]

    # Plotar exemplos corretos
    if len(correct) > 0:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(np.random.choice(correct, min(num_examples, len(correct)), replace=False)):
            if i >= num_examples:
                break

            plt.subplot(2, 5, i + 1)
            plt.imshow(X[idx])
            plt.title(f"Real: {'Vazia' if y[idx] == 0 else 'Ocupada'}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correct_examples.png'))
        plt.close()

    # Plotar exemplos incorretos
    if len(incorrect) > 0:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(np.random.choice(incorrect, min(num_examples, len(incorrect)), replace=False)):
            if i >= num_examples:
                break

            plt.subplot(2, 5, i + 1)
            plt.imshow(X[idx])
            plt.title(f"Real: {'Vazia' if y[idx] == 0 else 'Ocupada'}\nPred: {'Vazia' if y_pred[idx] == 0 else 'Ocupada'}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'incorrect_examples.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Treinar uma CNN para classificação de vagas de estacionamento')
    parser.add_argument('--input_dir', type=str, required=True, help='Diretório contendo as pastas empty e occupied')
    parser.add_argument('--output_dir', type=str, default='model_output', help='Diretório para salvar o modelo')
    parser.add_argument('--img_size', type=int, default=32, help='Tamanho para redimensionar as imagens')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do lote')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fração dos dados para teste')

    args = parser.parse_args()

    # Carregar dados
    X, y, filenames = load_data(args.input_dir, target_size=(args.img_size, args.img_size))

    # Dividir em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=args.test_split, stratify=y, random_state=42
    )

    # Treinar o modelo
    model, history = train_model(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        output_dir=args.output_dir
    )

    # Plotar histórico de treinamento
    plot_training_history(history, args.output_dir)

    # Avaliar o modelo
    eval_results = evaluate_model(model, X_test, y_test)

    # Plotar exemplos
    plot_examples(X_test, y_test, eval_results['y_pred'], filenames_test, args.output_dir)

    print(f"Treinamento concluído. Modelo salvo em {args.output_dir}")

if __name__ == "__main__":
    main()