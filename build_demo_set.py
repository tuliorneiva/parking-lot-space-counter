"""Seleciona 3 lotes x 3 ocupações (vazio/medio/cheio) com erros (<100%) para a demo."""
import glob, os, shutil, numpy as np, cv2, xml.etree.ElementTree as ET
from tensorflow.keras.models import load_model
GT={'space-empty':0,'space-occupied':1}
cnn=load_model('models/cnn.h5')

def parse(xml):
    r=[]
    for o in ET.parse(xml).getroot().findall('object'):
        n=o.findtext('name')
        if n not in GT: continue
        b=o.find('bndbox')
        r.append((int(float(b.findtext('xmin'))),int(float(b.findtext('ymin'))),
                  int(float(b.findtext('xmax'))),int(float(b.findtext('ymax'))),GT[n]))
    return r

def lot_of(n):
    if 20<=n<=33: return 'UFPR04'
    if 34<=n<=60: return 'UFPR05'
    if n>=80:     return 'PUCPR'
    return None

rows=[]
for xml in sorted(glob.glob('PKLot/test/*.xml')):
    jpg=xml[:-4]+'.jpg'; img=cv2.imread(jpg)
    if img is None: continue
    sp=parse(xml); lot=lot_of(len(sp))
    if not lot: continue
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X=np.empty((len(sp),32,32,3),np.float32); gt=[]
    for i,(a,b,c,d,g) in enumerate(sp):
        cr=rgb[b:d,a:c]; X[i]=cv2.resize(cr,(32,32))/255.0 if cr.size else 0; gt.append(g)
    pred=(cnn.predict(X,verbose=0).ravel()>=0.5).astype(int)
    acc=float((pred==np.array(gt)).mean()); occ=float(np.mean(gt))
    rows.append(dict(lot=lot,n=len(sp),acc=acc,occ=occ,jpg=jpg))

def bucket(occ):
    return 'vazio' if occ<0.34 else ('medio' if occ<0.67 else 'cheio')

# mapear lote -> estacN (ordem por tamanho)
lot_id={'UFPR04':'estac1','UFPR05':'estac2','PUCPR':'estac3'}
os.makedirs('demo_images',exist_ok=True)
# limpar antigos
for f in glob.glob('demo_images/estac*'): os.remove(f)

print(f"{len(rows)} imagens avaliadas\n")
for lot,eid in lot_id.items():
    for buck in ['vazio','medio','cheio']:
        cands=[r for r in rows if r['lot']==lot and bucket(r['occ'])==buck]
        if not cands:
            print(f"  {eid}-{buck} ({lot}): NENHUMA candidata"); continue
        # prefere acc<1.0 e com erros visiveis; escolhe a de menor acc >=0.85
        good=[r for r in cands if 0.85<=r['acc']<1.0]
        pick=min(good,key=lambda r:r['acc']) if good else min(cands,key=lambda r:r['acc'])
        dst=f"demo_images/{eid}-{buck}"
        shutil.copy(pick['jpg'],dst+'.jpg'); shutil.copy(pick['jpg'][:-4]+'.xml',dst+'.xml')
        print(f"  {eid}-{buck:5s} ({lot}): acc={pick['acc']*100:5.1f}%  n={pick['n']}  ocup={pick['occ']*100:3.0f}%")
print("\nPronto em demo_images/")
