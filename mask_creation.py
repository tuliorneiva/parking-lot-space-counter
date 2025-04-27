# perspective_parking_detector.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """Ordena os pontos para aplicar a transformação de perspectiva"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Soma das coordenadas
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Topo-esquerda
    rect[2] = pts[np.argmax(s)]  # Baixo-direita
    
    # Diferença entre coordenadas
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Topo-direita
    rect[3] = pts[np.argmax(diff)]  # Baixo-esquerda
    
    return rect

def four_point_transform(image, pts):
    """Aplica transformação de perspectiva para obter visão de cima"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular largura máxima
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calcular altura máxima
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Definir pontos de destino
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calcular matriz de transformação
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detect_parking_grid(warped_image, grid_size=(8, 4), margin=0.1):
    """Detecta uma grade de estacionamento na imagem transformada"""
    h, w = warped_image.shape[:2]
    
    # Calcular tamanho de cada célula da grade
    cell_width = w // grid_size[0]
    cell_height = h // grid_size[1]
    
    # Calcular margens
    margin_x = int(cell_width * margin)
    margin_y = int(cell_height * margin)
    
    # Criar retângulos para cada vaga
    parking_spaces = []
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            x1 = col * cell_width + margin_x
            y1 = row * cell_height + margin_y
            x2 = (col + 1) * cell_width - margin_x
            y2 = (row + 1) * cell_height - margin_y
            
            parking_spaces.append((x1, y1, x2, y2))
    
    return parking_spaces

def transform_spaces_to_original(spaces, M_inv, original_shape):
    """Transforma as coordenadas das vagas de volta para a imagem original"""
    original_spaces = []
    
    for x1, y1, x2, y2 in spaces:
        # Pontos da vaga na imagem transformada
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # Transformar de volta para a imagem original
        pts_transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M_inv)
        pts_transformed = pts_transformed.reshape(-1, 2)
        
        # Calcular bounding box
        x_min = max(0, int(np.min(pts_transformed[:, 0])))
        y_min = max(0, int(np.min(pts_transformed[:, 1])))
        x_max = min(original_shape[1], int(np.max(pts_transformed[:, 0])))
        y_max = min(original_shape[0], int(np.max(pts_transformed[:, 1])))
        
        original_spaces.append((x_min, y_min, x_max, y_max))
    
    return original_spaces

def main():
    # Carregar imagem
    img_path = 'PKLot/UFPR05/Sunny/2013-02-22/2013-02-22_06_05_00.jpg'
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return
    
    # Definir pontos para transformação de perspectiva
    # Estes pontos precisam ser ajustados manualmente para cada estacionamento
    pts = np.array([
        [100, 100],    # Topo-esquerda
        [image.shape[1] - 100, 100],    # Topo-direita
        [image.shape[1] - 100, image.shape[0] - 100],    # Baixo-direita
        [100, image.shape[0] - 100]     # Baixo-esquerda
    ], dtype=np.float32)
    
    # Aplicar transformação de perspectiva
    warped = four_point_transform(image, pts)
    
    # Calcular matriz inversa para transformação de volta
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [warped.shape[1] - 1, 0],
        [warped.shape[1] - 1, warped.shape[0] - 1],
        [0, warped.shape[0] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    
    # Detectar grade de estacionamento
    # Ajustar grid_size conforme o layout do estacionamento
    parking_spaces = detect_parking_grid(warped, grid_size=(8, 4))
    
    # Transformar vagas de volta para a imagem original
    original_spaces = transform_spaces_to_original(parking_spaces, M_inv, image.shape)
    
    # Visualizar as vagas na imagem transformada
    warped_vis = warped.copy()
    for i, (x1, y1, x2, y2) in enumerate(parking_spaces):
        cv2.rectangle(warped_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(warped_vis, str(i+1), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Visualizar as vagas na imagem original
    original_vis = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(original_spaces):
        cv2.rectangle(original_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_vis, str(i+1), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Criar máscaras para as vagas
    masks = []
    for x1, y1, x2, y2 in original_spaces:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        masks.append(mask)
    
    # Salvar as máscaras
    os.makedirs('parking_masks', exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(f'parking_masks/space_{i+1}.png', mask)
    
    # Criar uma máscara combinada
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    cv2.imwrite('parking_masks/combined_mask.png', combined_mask)
    
    # Exibir resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Transformada')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(warped_vis, cv2.COLOR_BGR2RGB))
    plt.title('Vagas na Imagem Transformada')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB))
    plt.title('Vagas na Imagem Original')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('parking_detection_results.png')
    plt.show()

if __name__ == "__main__":
    import os
    main()