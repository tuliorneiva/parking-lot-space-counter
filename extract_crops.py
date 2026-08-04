"""
extract_crops.py
Extrai recortes de vagas a partir do dataset PKLot (formato Pascal VOC do Roboflow)
e organiza em crops/<split>/<empty|occupied>/*.png.

Cada objeto <object> do XML vira um recorte da bounding box, redimensionado para um
tamanho fixo (default 64x64). Assim, cada modelo do benchmark reduz a partir daqui
para o tamanho que precisa (15x15 nos clássicos, 32x32 na CNN).

Uso:
    python extract_crops.py --src PKLot --out crops --size 64 \
        --max-per-class train=8000 valid=2000 test=2000
"""
import os
import glob
import argparse
import xml.etree.ElementTree as ET
import random
import cv2
from tqdm import tqdm

# nomes das classes no XML do Roboflow -> pasta de saída
CLASS_MAP = {
    'space-empty': 'empty',
    'space-occupied': 'occupied',
}


def parse_max_per_class(items):
    """Converte ['train=8000','valid=2000'] em {'train':8000,...}."""
    result = {}
    for it in items or []:
        split, val = it.split('=')
        result[split.strip()] = int(val)
    return result


def collect_objects(split_dir):
    """Retorna lista de (img_path, xml_class, bbox) para todas as vagas do split."""
    objects = []
    for xml_path in glob.glob(os.path.join(split_dir, '*.xml')):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        img_name = root.findtext('filename')
        img_path = os.path.join(split_dir, img_name)
        if not os.path.exists(img_path):
            # fallback: mesmo basename do xml
            img_path = xml_path[:-4] + '.jpg'
            if not os.path.exists(img_path):
                continue
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in CLASS_MAP:
                continue
            b = obj.find('bndbox')
            bbox = (
                int(float(b.findtext('xmin'))),
                int(float(b.findtext('ymin'))),
                int(float(b.findtext('xmax'))),
                int(float(b.findtext('ymax'))),
            )
            objects.append((img_path, CLASS_MAP[name], bbox))
    return objects


def balance_and_cap(objects, max_per_class, seed):
    """Embaralha e limita a no máximo max_per_class por classe (mantém balanceado)."""
    by_class = {'empty': [], 'occupied': []}
    for o in objects:
        by_class[o[1]].append(o)

    rng = random.Random(seed)
    for cls in by_class:
        rng.shuffle(by_class[cls])

    if max_per_class is None:
        # balanceia pela menor classe
        n = min(len(by_class['empty']), len(by_class['occupied']))
    else:
        n = min(max_per_class, len(by_class['empty']), len(by_class['occupied']))

    selected = by_class['empty'][:n] + by_class['occupied'][:n]
    rng.shuffle(selected)
    return selected, n


def extract_split(split, src, out, size, max_per_class, seed):
    split_dir = os.path.join(src, split)
    if not os.path.isdir(split_dir):
        print(f"  [pulando] {split_dir} não existe")
        return

    print(f"[{split}] lendo anotações...")
    objects = collect_objects(split_dir)
    selected, n = balance_and_cap(objects, max_per_class, seed)
    print(f"[{split}] {len(objects)} vagas -> usando {n}/classe ({len(selected)} recortes)")

    for cls in ('empty', 'occupied'):
        os.makedirs(os.path.join(out, split, cls), exist_ok=True)

    # cache da última imagem carregada (vagas da mesma imagem vêm agrupadas por arquivo)
    cache_path, cache_img = None, None
    counters = {'empty': 0, 'occupied': 0}

    for img_path, cls, (xmin, ymin, xmax, ymax) in tqdm(selected, desc=f"  recortando {split}"):
        if img_path != cache_path:
            cache_img = cv2.imread(img_path)
            cache_path = img_path
        if cache_img is None:
            continue
        h, w = cache_img.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        if xmax - xmin < 3 or ymax - ymin < 3:
            continue
        crop = cache_img[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop, (size, size))
        idx = counters[cls]
        counters[cls] += 1
        cv2.imwrite(os.path.join(out, split, cls, f"{cls}_{idx:06d}.png"), crop)

    print(f"[{split}] salvos -> empty={counters['empty']} occupied={counters['occupied']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='PKLot', help='raiz do dataset (com train/valid/test)')
    ap.add_argument('--out', default='crops', help='pasta de saída dos recortes')
    ap.add_argument('--size', type=int, default=64, help='tamanho do recorte quadrado')
    ap.add_argument('--max-per-class', nargs='*', default=None,
                    help='limite por classe por split, ex: train=8000 valid=2000 test=2000')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    caps = parse_max_per_class(args.max_per_class)
    for split in ('train', 'valid', 'test'):
        extract_split(split, args.src, args.out, args.size, caps.get(split), args.seed)

    print("\nConcluído. Recortes em:", os.path.abspath(args.out))


if __name__ == '__main__':
    main()
