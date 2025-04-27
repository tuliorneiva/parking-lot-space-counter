# Script para criar mscaras individuais para cada vaga de estacionamento
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import argparse
import json
import traceback

def create_individual_masks(img_shape, xml_path, output_dir):
    """
    Cria mscaras individuais para cada vaga de estacionamento.
    
    Args:
        img_shape: Dimenses da imagem (altura, largura)
        xml_path: Caminho para o arquivo XML com as coordenadas das vagas
        output_dir: Diretrio para salvar as mscaras
    
    Returns:
        Um dicionrio mapeando IDs de vagas para nomes de arquivos de mscara
    """
    try:
        # Criar o diretrio de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Carregar o XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Dicionrio para mapear IDs de vagas para nomes de arquivos
        mask_mapping = {}
        
        # Para cada vaga no XML
        for space in root.findall('.//space'):
            try:
                space_id = space.get('id')
                occupied = int(space.get('occupied'))
                
                # Extrair os pontos do contorno
                contour_points = []
                contour_element = space.find('contour')
                
                if contour_element is not None:
                    for point in contour_element.findall('point'):
                        x = point.get('x')
                        y = point.get('y')
                        
                        # Verificar se x e y no so None
                        if x is not None and y is not None:
                            contour_points.append([int(x), int(y)])
                    
                    # Verificar se temos pontos suficientes para criar um contorno
                    if len(contour_points) >= 3:
                        # Converter para array numpy
                        contour = np.array(contour_points)
                        
                        # Criar uma mscara vazia (preta)
                        mask = np.zeros(img_shape[:2], dtype=np.uint8)
                        
                        # Desenhar o contorno preenchido (branco) na mscara
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        
                        # Nome do arquivo da mscara
                        mask_filename = f"mask_space_{space_id}.png"
                        mask_path = os.path.join(output_dir, mask_filename)
                        
                        # Salvar a mscara
                        cv2.imwrite(mask_path, mask)
                        
                        # Adicionar ao mapeamento
                        mask_mapping[space_id] = {
                            'mask_file': mask_filename,
                            'occupied': occupied
                        }
            except Exception as e:
                print(f"Erro ao processar vaga {space_id}: {str(e)}")
                continue
        
        # Salvar o mapeamento em um arquivo JSON
        mapping_file = os.path.join(output_dir, "mask_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(mask_mapping, f, indent=4)
        
        print(f"Criadas {len(mask_mapping)} máscaras individuais em {output_dir}")
        print(f"Mapeamento salvo em {mapping_file}")
        
        return mask_mapping
    
    except Exception as e:
        print(f"Erro ao processar o arquivo XML: {str(e)}")
        traceback.print_exc()
        return None

def visualize_masks(img_path, masks_dir, mapping_file, output_path=None):
    """
    Visualiza a imagem original com as mscaras sobrepostas.
    
    Args:
        img_path: Caminho para a imagem original
        masks_dir: Diretrio contendo as mscaras
        mapping_file: Caminho para o arquivo de mapeamento
        output_path: Caminho para salvar a visualizao
    """
    # Carregar a imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return
    
    # Converter para RGB para visualizao
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Carregar o mapeamento
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Criar uma cpia da imagem para desenhar os contornos
    img_with_masks = img_rgb.copy()
    
    # Para cada vaga no mapeamento
    for space_id, info in mapping.items():
        # Carregar a mscara
        mask_path = os.path.join(masks_dir, info['mask_file'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Erro ao carregar a mscara: {mask_path}")
            continue
        
        # Encontrar os contornos da mscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"Nenhum contorno encontrado para a vaga {space_id}")
            continue
        
        # Cor baseada no status (verde para vazio, vermelho para ocupado)
        color = (0, 255, 0) if info['occupied'] == 0 else (255, 0, 0)
        
        # Desenhar o contorno
        cv2.drawContours(img_with_masks, contours, -1, color, 2)
        
        # Adicionar o ID da vaga
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img_with_masks, str(space_id), (cX, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mostrar a imagem
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_masks)
    plt.axis('off')
    plt.title('Vagas de Estacionamento')
    
    # Salvar a visualizao se um caminho for fornecido
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualizao salva em {output_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Criar mscaras individuais para vagas de estacionamento')
    parser.add_argument('--image', type=str, required=True, help='Caminho para a imagem do estacionamento')
    parser.add_argument('--xml', type=str, required=True, help='Caminho para o arquivo XML com as coordenadas')
    parser.add_argument('--output', type=str, default='masks', help='Diretrio para salvar as mscaras')
    parser.add_argument('--visualize', action='store_true', help='Visualizar as mscaras')
    parser.add_argument('--save_vis', type=str, default=None, help='Caminho para salvar a visualizao')
    
    args = parser.parse_args()
    
    # Carregar a imagem para obter suas dimenses
    img = cv2.imread(args.image)
    if img is None:
        print(f"Erro ao carregar a imagem: {args.image}")
        exit(1)
    
    # Criar as mscaras individuais
    mask_mapping = create_individual_masks(img.shape, args.xml, args.output)
    
    # Visualizar as mscaras se solicitado
    if args.visualize and mask_mapping:
        mapping_file = os.path.join(args.output, "mask_mapping.json")
        visualize_masks(args.image, args.output, mapping_file, args.save_vis)
