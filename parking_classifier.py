# parking_classifier.py
# Script para classificar vagas de estacionamento usando um modelo pré-treinado e máscaras individuais,
# comparando com ground truth do XML

import cv2
import numpy as np
import os
import json
import argparse
import pickle
from skimage.transform import resize
import matplotlib.pyplot as plt
from datetime import datetime
import xml.etree.ElementTree as ET
import traceback

def load_model(model_path):
    """
    Carrega o modelo de classificação.

    Args:
        model_path: Caminho para o arquivo do modelo

    Returns:
        O modelo carregado
    """
    try:
        model = pickle.load(open(model_path, "rb"))
        print(f"Modelo carregado de {model_path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        return None

def load_ground_truth(xml_path):
    """
    Carrega o status real (ground truth) das vagas a partir do arquivo XML.

    Args:
        xml_path: Caminho para o arquivo XML

    Returns:
        Um dicionário mapeando IDs de vagas para status (True se vazia, False se ocupada)
    """
    try:
        # Carregar o XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Dicionário para armazenar o status de cada vaga
        ground_truth = {}

        # Para cada vaga no XML
        for space in root.findall('.//space'):
            space_id = space.get('id')
            occupied_str = space.get('occupied')

            # Verificar se o atributo 'occupied' existe e não é None
            if occupied_str is not None:
                occupied = int(occupied_str)

                # True se vazia (0), False se ocupada (1)
                ground_truth[space_id] = occupied == 0

        print(f"Ground truth carregado de {xml_path}: {len(ground_truth)} vagas")
        return ground_truth

    except Exception as e:
        print(f"Erro ao carregar ground truth: {str(e)}")
        traceback.print_exc()
        return None

def classify_spot(img, mask, model):
    """
    Classifica uma vaga como vazia ou ocupada.

    Args:
        img: Imagem do estacionamento
        mask: Máscara da vaga
        model: Modelo de classificação

    Returns:
        True se a vaga estiver vazia, False se estiver ocupada
    """
    # Aplicar a máscara na imagem
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Encontrar a bounding box da máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Obter a bounding box do contorno
    x, y, w, h = cv2.boundingRect(contours[0])

    # Recortar a região da vaga
    spot_img = masked_img[y:y+h, x:x+w]

    # Se a imagem for muito pequena ou vazia, retornar None
    if spot_img.size == 0 or w < 5 or h < 5:
        return None

    # Redimensionar para o formato esperado pelo modelo (15x15)
    spot_resized = resize(spot_img, (15, 15))

    # Classificar
    prediction = model.predict([spot_resized.flatten()])[0]

    # Retorna True se vazia (0), False se ocupada (1)
    return prediction == 0

def calculate_metrics(predictions, ground_truth):
    """
    Calcula métricas de desempenho do modelo.

    Args:
        predictions: Dicionário com as previsões do modelo
        ground_truth: Dicionário com o status real das vagas

    Returns:
        Um dicionário com as métricas calculadas
    """
    # Inicializar contadores
    true_positives = 0  # Vaga vazia corretamente classificada como vazia
    true_negatives = 0  # Vaga ocupada corretamente classificada como ocupada
    false_positives = 0  # Vaga ocupada incorretamente classificada como vazia
    false_negatives = 0  # Vaga vazia incorretamente classificada como ocupada

    # Para cada vaga com ground truth
    for space_id, is_empty_gt in ground_truth.items():
        # Verificar se temos uma previsão para esta vaga
        if space_id in predictions:
            is_empty_pred = predictions[space_id]

            # Atualizar contadores
            if is_empty_gt and is_empty_pred:
                true_positives += 1
            elif not is_empty_gt and not is_empty_pred:
                true_negatives += 1
            elif not is_empty_gt and is_empty_pred:
                false_positives += 1
            elif is_empty_gt and not is_empty_pred:
                false_negatives += 1

    # Calcular métricas
    total = true_positives + true_negatives + false_positives + false_negatives

    if total == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total': 0
        }

    accuracy = (true_positives + true_negatives) / total

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total': total
    }

def process_parking_lot(img_path, masks_dir, mapping_file, model, xml_path=None, output_dir=None, lot_name=None):
    """
    Processa um estacionamento, classificando todas as vagas.

    Args:
        img_path: Caminho para a imagem do estacionamento
        masks_dir: Diretório contendo as máscaras individuais
        mapping_file: Caminho para o arquivo de mapeamento
        model: Modelo de classificação
        xml_path: Caminho para o arquivo XML com ground truth (opcional)
        output_dir: Diretório para salvar os resultados (opcional)
        lot_name: Nome do estacionamento (opcional)

    Returns:
        Um dicionário com o status de cada vaga e a imagem com os resultados
    """
    # Carregar a imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return None, None, None

    # Carregar o mapeamento
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    # Carregar ground truth se disponível
    ground_truth = None
    if xml_path:
        ground_truth = load_ground_truth(xml_path)

    # Dicionário para armazenar o status de cada vaga
    spots_status = {}

    # Criar uma cópia da imagem para visualização
    img_result = img.copy()

    # Para cada vaga no mapeamento
    for space_id, info in mapping.items():
        # Carregar a máscara
        mask_path = os.path.join(masks_dir, info['mask_file'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Erro ao carregar a máscara: {mask_path}")
            continue

        # Classificar a vaga
        is_empty = classify_spot(img, mask, model)

        # Se a classificação falhar, usar o status do XML (se disponível)
        if is_empty is None:
            if 'occupied' in info:
                is_empty = info['occupied'] == 0
                print(f"Falha na classificação da vaga {space_id}, usando status do XML: {'vazia' if is_empty else 'ocupada'}")
            else:
                # Se não tiver informação, assumir ocupada
                is_empty = False
                print(f"Falha na classificação da vaga {space_id}, assumindo ocupada")

        # Armazenar o status
        spots_status[space_id] = is_empty

        # Encontrar os contornos da máscara para visualização
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Verificar se a previsão está correta (se tivermos ground truth)
            is_correct = None
            if ground_truth and space_id in ground_truth:
                is_correct = spots_status[space_id] == ground_truth[space_id]

            # Definir a cor com base no status e na correção
            if ground_truth and space_id in ground_truth:
                if is_correct:
                    # Previsão correta: verde para vazia, vermelho para ocupada
                    color = (0, 255, 0) if spots_status[space_id] else (0, 0, 255)
                else:
                    # Previsão incorreta: amarelo
                    color = (0, 255, 255)
            else:
                # Sem ground truth: verde para vazia, vermelho para ocupada
                color = (0, 255, 0) if is_empty else (0, 0, 255)

            # Desenhar o contorno
            cv2.drawContours(img_result, contours, -1, color, 2)

            # Adicionar o ID da vaga
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img_result, str(space_id), (cX, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calcular métricas se tivermos ground truth
    metrics = None
    if ground_truth:
        metrics = calculate_metrics(spots_status, ground_truth)
        print(f"Métricas de desempenho:")
        print(f"  Acuracia: {metrics['accuracy']:.4f}")
        print(f"  Precisão: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Verdadeiros Positivos: {metrics['true_positives']}")
        print(f"  Verdadeiros Negativos: {metrics['true_negatives']}")
        print(f"  Falsos Positivos: {metrics['false_positives']}")
        print(f"  Falsos Negativos: {metrics['false_negatives']}")

    # Contar vagas vazias
    empty_count = sum(1 for status in spots_status.values() if status)
    total_count = len(spots_status)

    # Adicionar texto com contagem
    cv2.rectangle(img_result, (10, 10), (400, 130 if metrics else 70), (0, 0, 0), -1)

    # Adicionar nome do estacionamento se fornecido
    y_pos = 30
    if lot_name:
        cv2.putText(img_result, f'Estacionamento: {lot_name}', 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

    # Adicionar contagem de vagas
    cv2.putText(img_result, f'Vagas disponiveis: {empty_count} / {total_count}', 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30

    # Adicionar métricas se disponíveis
    if metrics:
        cv2.putText(img_result, f'Acurácia: {metrics["accuracy"]:.4f}', 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

    # Adicionar legenda
    if ground_truth:
        # Adicionar legenda para as cores
        legend_x = img.shape[1] - 200
        legend_y = 70

        # Vaga vazia (verde)
        cv2.rectangle(img_result, (legend_x, legend_y), (legend_x + 20, legend_y + 20), (0, 255, 0), -1)
        cv2.putText(img_result, "Vazia", (legend_x + 30, legend_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Vaga ocupada (vermelho)
        cv2.rectangle(img_result, (legend_x, legend_y + 30), (legend_x + 20, legend_y + 50), (0, 0, 255), -1)
        cv2.putText(img_result, "Ocupada", (legend_x + 30, legend_y + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Previsão incorreta (amarelo)
        cv2.rectangle(img_result, (legend_x, legend_y + 60), (legend_x + 20, legend_y + 80), (0, 255, 255), -1)
        cv2.putText(img_result, "Incorreta", (legend_x + 30, legend_y + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Adicionar timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img_result, timestamp, 
                (img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Salvar a imagem com os resultados se um diretório for fornecido
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Criar nome de arquivo baseado no timestamp
        filename = f"{lot_name or 'parking'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = os.path.join(output_dir, filename)

        # Salvar a imagem
        cv2.imwrite(output_path, img_result)
        print(f"Resultado salvo em {output_path}")

        # Salvar também os dados de status e métricas em JSON
        status_filename = f"{lot_name or 'parking'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_status.json"
        status_path = os.path.join(output_dir, status_filename)

        # Converter valores booleanos para inteiros para evitar problemas de serialização
        serializable_spots_status = {k: 1 if v else 0 for k, v in spots_status.items()}

        status_data = {
            'timestamp': timestamp,
            'lot_name': lot_name,
            'total_spots': total_count,
            'empty_spots': empty_count,
            'spots_status': serializable_spots_status
        }

        # Adicionar métricas se disponíveis
        if metrics:
            status_data['metrics'] = metrics

        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=4)

        print(f"Dados de status salvos em {status_path}")

    return spots_status, img_result, metrics

def process_multiple_images(image_dir, masks_dir, mapping_file, model, xml_dir=None, output_dir=None, lot_name=None):
    """
    Processa múltiplas imagens de um estacionamento.

    Args:
        image_dir: Diretório contendo as imagens do estacionamento
        masks_dir: Diretório contendo as máscaras individuais
        mapping_file: Caminho para o arquivo de mapeamento
        model: Modelo de classificação
        xml_dir: Diretório contendo os arquivos XML com ground truth (opcional)
        output_dir: Diretório para salvar os resultados (opcional)
        lot_name: Nome do estacionamento (opcional)
    """
    # Listar todas as imagens no diretório
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"Nenhuma imagem encontrada em {image_dir}")
        return

    print(f"Processando {len(image_files)} imagens de {image_dir}...")

    # Métricas acumuladas
    all_metrics = []

    # Processar cada imagem
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"Processando {img_path}...")

        # Verificar se existe um arquivo XML correspondente
        xml_path = None
        if xml_dir:
            xml_file = os.path.splitext(img_file)[0] + '.xml'
            xml_path_candidate = os.path.join(xml_dir, xml_file)
            if os.path.exists(xml_path_candidate):
                xml_path = xml_path_candidate
            else:
                # Tentar no mesmo diretório da imagem
                xml_path_candidate = os.path.join(image_dir, xml_file)
                if os.path.exists(xml_path_candidate):
                    xml_path = xml_path_candidate

        # Processar o estacionamento
        spots_status, img_result, metrics = process_parking_lot(
            img_path, masks_dir, mapping_file, model, xml_path, output_dir, lot_name
        )

        if spots_status is None:
            print(f"Falha ao processar {img_path}")
            continue

        # Adicionar métricas à lista
        if metrics:
            all_metrics.append(metrics)

        # Mostrar a imagem com os resultados
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Status das Vagas - {lot_name or "Estacionamento"} - {os.path.basename(img_path)}')
        plt.show()

    # Calcular métricas médias
    if all_metrics:
        avg_accuracy = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
        avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['f1_score'] for m in all_metrics) / len(all_metrics)

        print(f"Métricas médias para {len(all_metrics)} imagens:")
        print(f"  Acurácia média: {avg_accuracy:.4f}")
        print(f"  Precisão média: {avg_precision:.4f}")
        print(f"  Recall médio: {avg_recall:.4f}")
        print(f"  F1-Score médio: {avg_f1:.4f}")

        # Salvar métricas médias em JSON
        if output_dir:
            metrics_filename = f"{lot_name or 'parking'}_avg_metrics.json"
            metrics_path = os.path.join(output_dir, metrics_filename)

            with open(metrics_path, 'w') as f:
                json.dump({
                    'avg_accuracy': float(avg_accuracy),
                    'avg_precision': float(avg_precision),
                    'avg_recall': float(avg_recall),
                    'avg_f1_score': float(avg_f1),
                    'num_images': len(all_metrics)
                }, f, indent=4)

            print(f"Métricas médias salvas em {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='Classificar vagas de estacionamento usando modelo pré-treinado')

    # Argumentos obrigatórios
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo de classificação')

    # Grupo de argumentos para processamento de imagem única
    single_group = parser.add_argument_group('Processamento de imagem única')
    single_group.add_argument('--image', type=str, help='Caminho para a imagem do estacionamento')
    single_group.add_argument('--xml', type=str, help='Caminho para o arquivo XML com ground truth')

    # Grupo de argumentos para processamento de múltiplas imagens
    multi_group = parser.add_argument_group('Processamento de múltiplas imagens')
    multi_group.add_argument('--image_dir', type=str, help='Diretório contendo imagens do estacionamento')
    multi_group.add_argument('--xml_dir', type=str, help='Diretório contendo arquivos XML com ground truth')

    # Argumentos comuns
    parser.add_argument('--masks', type=str, help='Diretório contendo as máscaras individuais')
    parser.add_argument('--mapping', type=str, help='Caminho para o arquivo de mapeamento (padrão: masks_dir/mask_mapping.json)')
    parser.add_argument('--output', type=str, help='Diretório para salvar os resultados')
    parser.add_argument('--lot_name', type=str, help='Nome do estacionamento (UFPR04, UFPR05, PUC)')

    args = parser.parse_args()

    # Verificar se pelo menos uma opção de imagem foi fornecida
    if not args.image and not args.image_dir:
        parser.error("É necessário fornecer --image ou --image_dir")

    # Verificar se o diretório de máscaras foi fornecido
    if not args.masks:
        parser.error("É necessário fornecer --masks")

    # Definir o caminho do mapeamento se não for fornecido
    if not args.mapping:
        args.mapping = os.path.join(args.masks, "mask_mapping.json")

    # Carregar o modelo
    model = load_model(args.model)
    if model is None:
        return

    # Processar imagem única ou múltiplas imagens
    if args.image:
        # Processar uma única imagem
        spots_status, img_result, metrics = process_parking_lot(
            args.image, args.masks, args.mapping, model, args.xml, args.output, args.lot_name
        )

        if spots_status is not None:
            # Mostrar a imagem com os resultados
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Status das Vagas - {args.lot_name or "Estacionamento"}')
            plt.show()

    elif args.image_dir:
        # Processar múltiplas imagens
        process_multiple_images(
            args.image_dir, args.masks, args.mapping, model, args.xml_dir, args.output, args.lot_name
        )

if __name__ == "__main__":
    main()
