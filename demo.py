"""
demo.py — DEMONSTRAÇÃO
Roda um modelo (ou todos) numa imagem de estacionamento anotada (formato Pascal VOC do
Roboflow), classifica cada vaga como vazia/ocupada, desenha o resultado e compara com o
gabarito da anotação.

  verde   = vaga livre (classificada certo)
  vermelho= vaga ocupada (classificada certo)
  amarelo = o modelo ERROU aquela vaga

Uso:
    python demo.py --image demo_images/estac1.jpg --model cnn
    python demo.py --image demo_images/estac1.jpg --model all
"""
import os, sys, pickle, argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2

GT = {'space-empty': 0, 'space-occupied': 1}   # 0 = vazia, 1 = ocupada
MODEL_NAMES = {'svm': 'SVM', 'knn': 'KNN', 'rf': 'Random Forest', 'cnn': 'CNN'}


def parse_xml(xml_path):
    """Devolve lista de (xmin,ymin,xmax,ymax, gabarito 0/1)."""
    spots = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        name = obj.findtext('name')
        if name not in GT:
            continue
        b = obj.find('bndbox')
        spots.append((int(float(b.findtext('xmin'))), int(float(b.findtext('ymin'))),
                      int(float(b.findtext('xmax'))), int(float(b.findtext('ymax'))),
                      GT[name]))
    return spots


def load_model(key):
    if key == 'cnn':
        from tensorflow.keras.models import load_model as lm
        return lm('models/cnn.h5')
    return pickle.load(open(f'models/{key}.pkl', 'rb'))


def predict_spot(model, key, img_rgb, box):
    xmin, ymin, xmax, ymax = box
    crop = img_rgb[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return 1
    if key == 'cnn':
        x = cv2.resize(crop, (32, 32)).astype('float32') / 255.0
        return int(model.predict(x[None], verbose=0).ravel()[0] >= 0.5)
    x = (cv2.resize(crop, (15, 15)).astype('float32') / 255.0).flatten()
    return int(model.predict([x])[0])


def run_model(key, img_bgr, img_rgb, spots):
    model = load_model(key)
    out = img_bgr.copy()
    correct = free = 0
    for (xmin, ymin, xmax, ymax, gt) in spots:
        pred = predict_spot(model, key, img_rgb, (xmin, ymin, xmax, ymax))
        ok = (pred == gt)
        correct += ok
        free += (pred == 0)
        if not ok:
            color = (0, 255, 255)                       # amarelo = erro
        else:
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # verde/vermelho
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)

    total = len(spots)
    acc = correct / total if total else 0
    # painel de texto
    cv2.rectangle(out, (8, 8), (360, 96), (0, 0, 0), -1)
    cv2.putText(out, f'Modelo: {MODEL_NAMES[key]}', (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f'Vagas livres: {free} / {total}', (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(out, f'Acuracia: {acc*100:.1f}%', (16, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out, free, total, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='imagem (com .xml de mesmo nome ao lado)')
    ap.add_argument('--model', default='cnn', choices=['svm', 'knn', 'rf', 'cnn', 'all'])
    ap.add_argument('--out', default='demo_results')
    args = ap.parse_args()

    xml_path = os.path.splitext(args.image)[0] + '.xml'
    if not os.path.exists(xml_path):
        sys.exit(f"XML não encontrado: {xml_path}")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        sys.exit(f"Não consegui abrir a imagem: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    spots = parse_xml(xml_path)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]

    keys = ['svm', 'knn', 'rf', 'cnn'] if args.model == 'all' else [args.model]
    print(f"\nImagem: {args.image}  ({len(spots)} vagas anotadas)")
    print("-" * 52)
    for key in keys:
        out, free, total, acc = run_model(key, img_bgr, img_rgb, spots)
        path = os.path.join(args.out, f'{base}__{key}.jpg')
        cv2.imwrite(path, out)
        print(f"  {MODEL_NAMES[key]:14s} livres={free:2d}/{total}  "
              f"acuracia={acc*100:5.1f}%  ->  {path}")
    print("-" * 52)
    print("Abra as imagens em", os.path.abspath(args.out))


if __name__ == '__main__':
    main()
