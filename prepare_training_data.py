# prepare_training_data_balanced.py
# Script para preparar dados de treinamento com distribuição equilibrada entre estacionamentos

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import shutil
import random
from collections import defaultdict

def extract_parking_spots(img_path, xml_path, output_dir, lot_name=None, max_per_image=None):
    """
    Extrai as vagas de estacionamento de uma imagem usando as coordenadas do XML.

    Args:
        img_path: Caminho para a imagem
        xml_path: Caminho para o arquivo XML
        output_dir: Diretório para salvar as vagas extraídas
        lot_name: Nome do estacionamento (opcional)
        max_per_image: Número máximo de vagas a extrair por imagem (opcional)

    Returns:
        Número de vagas vazias e ocupadas extraídas
    """
    # Carregar a imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return 0, 0

    # Carregar o XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Erro ao carregar o XML {xml_path}: {str(e)}")
        return 0, 0

    # Criar diretórios para vagas vazias e ocupadas
    empty_dir = os.path.join(output_dir, 'empty')
    occupied_dir = os.path.join(output_dir, 'occupied')

    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(occupied_dir, exist_ok=True)

    # Coletar todas as vagas
    empty_spots = []
    occupied_spots = []

    # Para cada vaga no XML
    for space in root.findall('.//space'):
        space_id = space.get('id')

        # Verificar se o atributo 'occupied' existe
        occupied_str = space.get('occupied')
        if occupied_str is None:
            continue

        # Converter para inteiro
        occupied = int(occupied_str)

        # Obter os pontos do contorno
        contour_points = []
        for point in space.findall('.//point'):
            x = int(float(point.get('x')))
            y = int(float(point.get('y')))
            contour_points.append([x, y])

        if not contour_points:
            continue

        # Converter para array numpy
        contour = np.array(contour_points)

        # Criar uma máscara para a vaga
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Aplicar a máscara na imagem
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Encontrar a bounding box do contorno
        x, y, w, h = cv2.boundingRect(contour)

        # Recortar a região da vaga
        spot_img = masked_img[y:y+h, x:x+w]

        # Se a imagem for muito pequena, pular
        if spot_img.size == 0 or w < 5 or h < 5:
            continue

        # Criar nome de arquivo único
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if lot_name:
            filename = f"{lot_name}_{base_name}_spot_{space_id}.png"
        else:
            filename = f"{base_name}_spot_{space_id}.png"

        # Adicionar à lista apropriada
        if occupied == 0:  # Vaga vazia
            empty_spots.append((spot_img, filename))
        else:  # Vaga ocupada
            occupied_spots.append((spot_img, filename))

    # Limitar o número de vagas por imagem, se especificado
    if max_per_image is not None:
        # Limitar vagas vazias
        if len(empty_spots) > max_per_image:
            empty_spots = random.sample(empty_spots, max_per_image)

        # Limitar vagas ocupadas
        if len(occupied_spots) > max_per_image:
            occupied_spots = random.sample(occupied_spots, max_per_image)

    # Salvar as vagas
    for spot_img, filename in empty_spots:
        output_path = os.path.join(empty_dir, filename)
        cv2.imwrite(output_path, spot_img)

    for spot_img, filename in occupied_spots:
        output_path = os.path.join(occupied_dir, filename)
        cv2.imwrite(output_path, spot_img)

    return len(empty_spots), len(occupied_spots)

def process_parking_lot(img_dir, xml_dir, output_dir, lot_name=None, max_per_lot=None, max_per_image=None):
    """
    Processa todas as imagens de um estacionamento.

    Args:
        img_dir: Diretório contendo as imagens
        xml_dir: Diretório contendo os arquivos XML
        output_dir: Diretório para salvar as vagas extraídas
        lot_name: Nome do estacionamento (opcional)
        max_per_lot: Número máximo de vagas a extrair por estacionamento (opcional)
        max_per_image: Número máximo de vagas a extrair por imagem (opcional)
    """
    # Listar todos os arquivos de imagem
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not img_files:
        print(f"Nenhuma imagem encontrada em {img_dir}")
        return 0, 0

    print(f"Processando {len(img_files)} imagens de {img_dir}...")

    # Contadores totais
    total_empty = 0
    total_occupied = 0

    # Se temos um limite por lote, selecionar aleatoriamente as imagens
    if max_per_lot is not None:
        # Embaralhar as imagens para garantir aleatoriedade
        random.shuffle(img_files)

    # Processar cada imagem
    for img_file in tqdm(img_files):
        # Caminho da imagem
        img_path = os.path.join(img_dir, img_file)

        # Caminho do XML correspondente
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(xml_dir, xml_file)

        # Verificar se o XML existe
        if not os.path.exists(xml_path):
            print(f"XML não encontrado para {img_file}")
            continue

        # Extrair as vagas
        empty_count, occupied_count = extract_parking_spots(
            img_path, xml_path, output_dir, lot_name, max_per_image
        )

        # Atualizar contadores
        total_empty += empty_count
        total_occupied += occupied_count

        # Verificar se atingimos o limite por lote
        if max_per_lot is not None:
            if total_empty + total_occupied >= max_per_lot:
                print(f"Limite de {max_per_lot} vagas atingido para {lot_name}")
                break

    print(f"Processamento concluído para {lot_name or 'estacionamento'}:")
    print(f"  Vagas vazias extraídas: {total_empty}")
    print(f"  Vagas ocupadas extraídas: {total_occupied}")
    print(f"  Total de vagas extraídas: {total_empty + total_occupied}")

    return total_empty, total_occupied

def balance_dataset(output_dir, max_samples=None, lot_stats=None):
    """
    Equilibra o conjunto de dados, garantindo o mesmo número de amostras em cada classe.

    Args:
        output_dir: Diretório contendo as pastas 'empty' e 'occupied'
        max_samples: Número máximo de amostras por classe (opcional)
        lot_stats: Estatísticas por estacionamento (opcional)
    """
    # Caminhos para os diretórios
    empty_dir = os.path.join(output_dir, 'empty')
    occupied_dir = os.path.join(output_dir, 'occupied')

    # Verificar se os diretórios existem
    if not os.path.exists(empty_dir) or not os.path.exists(occupied_dir):
        print("Diretórios 'empty' ou 'occupied' não encontrados")
        return

    # Listar os arquivos
    empty_files = os.listdir(empty_dir)
    occupied_files = os.listdir(occupied_dir)

    print(f"Total de vagas vazias: {len(empty_files)}")
    print(f"Total de vagas ocupadas: {len(occupied_files)}")

    # Se temos estatísticas por estacionamento, mostrar a distribuição
    if lot_stats:
        print("\nDistribuição por estacionamento:")
        for lot, stats in lot_stats.items():
            print(f"  {lot}: {stats['empty']} vazias, {stats['occupied']} ocupadas")

    # Determinar o número de amostras por classe
    if max_samples:
        samples_per_class = min(len(empty_files), len(occupied_files), max_samples)
    else:
        samples_per_class = min(len(empty_files), len(occupied_files))

    print(f"\nEquilibrando o conjunto de dados para {samples_per_class} amostras por classe...")

    # Função para equilibrar uma classe
    def balance_class(files, directory, target_count):
        if len(files) > target_count:
            # Agrupar arquivos por estacionamento
            lot_files = defaultdict(list)
            for file in files:
                # Extrair o nome do estacionamento do nome do arquivo
                parts = file.split('_')
                if len(parts) >= 2:
                    lot = parts[0]
                    lot_files[lot].append(file)
                else:
                    # Se não conseguir extrair, colocar em "outros"
                    lot_files["outros"].append(file)

            # Calcular quantos arquivos selecionar de cada estacionamento
            total_lots = len(lot_files)
            if total_lots == 0:
                return

            # Distribuir igualmente entre os estacionamentos
            files_per_lot = target_count // total_lots

            # Selecionar arquivos de cada estacionamento
            selected_files = []
            for lot, lot_file_list in lot_files.items():
                # Limitar ao número disponível
                count = min(files_per_lot, len(lot_file_list))
                selected_files.extend(random.sample(lot_file_list, count))

            # Se ainda não atingimos o alvo, adicionar mais arquivos aleatoriamente
            remaining = target_count - len(selected_files)
            if remaining > 0:
                # Criar uma lista de todos os arquivos que não foram selecionados
                remaining_files = [f for f in files if f not in selected_files]
                if remaining_files:
                    # Adicionar arquivos aleatórios até atingir o alvo
                    additional = random.sample(remaining_files, min(remaining, len(remaining_files)))
                    selected_files.extend(additional)

            # Remover os arquivos que não foram selecionados
            for file in files:
                if file not in selected_files:
                    os.remove(os.path.join(directory, file))

    # Equilibrar cada classe
    balance_class(empty_files, empty_dir, samples_per_class)
    balance_class(occupied_files, occupied_dir, samples_per_class)

    # Verificar o resultado
    empty_files = os.listdir(empty_dir)
    occupied_files = os.listdir(occupied_dir)

    print(f"Conjunto de dados equilibrado:")
    print(f"  Vagas vazias: {len(empty_files)}")
    print(f"  Vagas ocupadas: {len(occupied_files)}")

def main():
    parser = argparse.ArgumentParser(description='Preparar dados de treinamento para classificação de vagas')

    # Argumentos para processamento de um único estacionamento
    parser.add_argument('--img_dir', type=str, help='Diretório contendo as imagens do estacionamento')
    parser.add_argument('--xml_dir', type=str, help='Diretório contendo os arquivos XML')
    parser.add_argument('--lot_name', type=str, help='Nome do estacionamento (UFPR04, UFPR05, PUC)')

    # Argumentos para processamento de múltiplos estacionamentos
    parser.add_argument('--pklot_dir', type=str, help='Diretório raiz do PKLot (contendo PUCPR, UFPR04, UFPR05)')

    # Argumentos comuns
    parser.add_argument('--output_dir', type=str, required=True, help='Diretório para salvar as vagas extraídas')
    parser.add_argument('--max_samples', type=int, help='Número máximo de amostras por classe')
    parser.add_argument('--max_per_lot', type=int, help='Número máximo de vagas a extrair por estacionamento')
    parser.add_argument('--max_per_image', type=int, help='Número máximo de vagas a extrair por imagem')
    parser.add_argument('--balance', action='store_true', help='Equilibrar o conjunto de dados')

    args = parser.parse_args()

    # Verificar se pelo menos uma opção foi fornecida
    if not args.img_dir and not args.pklot_dir:
        parser.error("É necessário fornecer --img_dir ou --pklot_dir")

    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)

    # Estatísticas por estacionamento
    lot_stats = defaultdict(lambda: {'empty': 0, 'occupied': 0})

    # Processar um único estacionamento
    if args.img_dir and args.xml_dir:
        empty_count, occupied_count = process_parking_lot(
            args.img_dir, args.xml_dir, args.output_dir, args.lot_name,
            args.max_per_lot, args.max_per_image
        )
        if args.lot_name:
            lot_stats[args.lot_name]['empty'] += empty_count
            lot_stats[args.lot_name]['occupied'] += occupied_count

    # Processar múltiplos estacionamentos do PKLot
    if args.pklot_dir:
        # Estacionamentos do PKLot
        lots = ['PUCPR', 'UFPR04', 'UFPR05']

        # Condições climáticas
        conditions = ['Cloudy', 'Rainy', 'Sunny']

        # Calcular o número máximo de vagas por estacionamento/condição
        max_per_condition = None
        if args.max_per_lot and args.max_samples:
            # Distribuir igualmente entre estacionamentos e condições
            total_combinations = len(lots) * len(conditions)
            max_per_condition = args.max_per_lot // total_combinations

        # Processar cada estacionamento e condição
        for lot in lots:
            lot_dir = os.path.join(args.pklot_dir, lot)
            if not os.path.exists(lot_dir):
                print(f"Diretório não encontrado: {lot_dir}")
                continue

            for condition in conditions:
                # Diretório da condição
                condition_dir = os.path.join(lot_dir, condition)

                if not os.path.exists(condition_dir):
                    print(f"Diretório não encontrado: {condition_dir}")
                    continue

                # Listar as datas
                dates = [d for d in os.listdir(condition_dir) if os.path.isdir(os.path.join(condition_dir, d))]

                # Embaralhar as datas para garantir aleatoriedade
                random.shuffle(dates)

                # Contador para este estacionamento/condição
                condition_empty = 0
                condition_occupied = 0

                for date in dates:
                    # Diretório da data
                    date_dir = os.path.join(condition_dir, date)

                    # Nome do lote
                    lot_condition_name = f"{lot}_{condition}"

                    # Processar o estacionamento
                    print(f"Processando {lot} - {condition} - {date}...")

                    # Calcular o máximo restante para esta condição
                    remaining_max = None
                    if max_per_condition is not None:
                        remaining_max = max_per_condition - (condition_empty + condition_occupied)
                        if remaining_max <= 0:
                            print(f"Limite atingido para {lot_condition_name}")
                            break

                    empty_count, occupied_count = process_parking_lot(
                        date_dir, date_dir, args.output_dir, lot_condition_name,
                        remaining_max, args.max_per_image
                    )

                    # Atualizar contadores
                    condition_empty += empty_count
                    condition_occupied += occupied_count
                    lot_stats[lot_condition_name]['empty'] += empty_count
                    lot_stats[lot_condition_name]['occupied'] += occupied_count

                    # Verificar se atingimos o limite para esta condição
                    if max_per_condition is not None:
                        if condition_empty + condition_occupied >= max_per_condition:
                            print(f"Limite de {max_per_condition} vagas atingido para {lot_condition_name}")
                            break

    # Equilibrar o conjunto de dados se solicitado
    if args.balance:
        balance_dataset(args.output_dir, args.max_samples, lot_stats)

if __name__ == "__main__":
    main()