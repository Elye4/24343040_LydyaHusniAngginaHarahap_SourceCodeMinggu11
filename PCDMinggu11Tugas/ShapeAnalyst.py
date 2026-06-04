"""
Shape Analysis Pipeline untuk Klasifikasi Objek
Implementasi lengkap: region properties, moments, chain codes, Fourier descriptors, k-NN
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os, json

os.makedirs('/home/claude/output', exist_ok=True)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
def make_circle(cx, cy, r, size=128, noise=0):
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (cx, cy), r, 255, -1)
    if noise:
        n = np.random.randint(0, noise, img.shape, np.uint8)
        img = cv2.add(img, n)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

def make_rect(cx, cy, w, h, angle=0, size=128):
    img = np.zeros((size, size), np.uint8)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    box = np.intp(box)
    cv2.fillPoly(img, [box], 255)
    return img

def make_triangle(cx, cy, r, angle=0, size=128):
    img = np.zeros((size, size), np.uint8)
    pts = []
    for i in range(3):
        a = np.radians(angle + i * 120)
        pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

def generate_dataset():
    np.random.seed(42)
    dataset = {}
    
    # Kelas 1: CIRCLE (variasi ukuran, posisi)
    circles = []
    params_c = [(64,64,25),(64,64,35),(64,64,20),(55,70,28),(70,55,32),
                (64,64,18),(60,65,30),(68,60,22),(64,64,40),(58,72,26)]
    for cx,cy,r in params_c:
        circles.append(make_circle(cx,cy,r))
    dataset['circle'] = circles

    # Kelas 2: RECTANGLE (variasi ukuran, rotasi)
    rects = []
    params_r = [(64,64,50,30,0),(64,64,60,35,15),(64,64,45,25,30),
                (64,64,55,40,45),(64,64,40,20,0),(64,64,50,30,60),
                (60,68,55,28,10),(68,60,48,32,20),(64,64,65,30,0),(64,64,42,42,0)]
    for cx,cy,w,h,a in params_r:
        rects.append(make_rect(cx,cy,w,h,a))
    dataset['rectangle'] = rects

    # Kelas 3: TRIANGLE (variasi rotasi, ukuran)
    tris = []
    params_t = [(64,64,35,0),(64,64,30,30),(64,64,40,60),(64,64,32,90),
                (64,64,38,120),(64,64,28,45),(62,66,35,15),(66,62,33,75),
                (64,64,42,150),(64,64,27,180)]
    for cx,cy,r,a in params_t:
        tris.append(make_triangle(cx,cy,r,a))
    dataset['triangle'] = tris

    return dataset

# ─────────────────────────────────────────────
# 2. REGION PROPERTIES
# ─────────────────────────────────────────────
def extract_region_properties(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    cx = M['m10']/M['m00'] if M['m00'] else 0
    cy = M['m01']/M['m00'] if M['m00'] else 0
    x,y,w,h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    aspect_ratio = float(w)/h if h else 0
    extent = area/(w*h) if w*h else 0
    solidity = area/hull_area if hull_area else 0
    equiv_diam = np.sqrt(4*area/np.pi) if area else 0
    circularity = (4*np.pi*area)/(perimeter**2) if perimeter else 0
    return {
        'area': area, 'perimeter': perimeter,
        'centroid_x': cx, 'centroid_y': cy,
        'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
        'hull_area': hull_area, 'aspect_ratio': aspect_ratio,
        'extent': extent, 'solidity': solidity,
        'equiv_diameter': equiv_diam, 'circularity': circularity,
        'contour': cnt
    }

# ─────────────────────────────────────────────
# 3. MOMENTS
# ─────────────────────────────────────────────
def compute_moments(mask):
    M = cv2.moments(mask)
    # Spatial moments
    m00, m10, m01 = M['m00'], M['m10'], M['m01']
    # Central moments
    mu20, mu02, mu11 = M['mu20'], M['mu02'], M['mu11']
    mu30, mu03, mu21, mu12 = M['mu30'], M['mu03'], M['mu21'], M['mu12']
    # Normalized central moments
    if m00 > 0:
        nu20 = mu20/m00**2; nu02 = mu02/m00**2; nu11 = mu11/m00**2
        nu30 = mu30/m00**2.5; nu03 = mu03/m00**2.5
        nu21 = mu21/m00**2.5; nu12 = mu12/m00**2.5
    else:
        nu20=nu02=nu11=nu30=nu03=nu21=nu12=0
    # Hu moments
    hu = cv2.HuMoments(M).flatten()
    # Sign-based log transform
    hu_log = np.zeros(7)
    for i, h in enumerate(hu):
        if h != 0:
            hu_log[i] = -np.sign(h) * np.log10(abs(h))
    return {
        'm00': m00, 'm10': m10, 'm01': m01,
        'mu20': mu20, 'mu02': mu02, 'mu11': mu11,
        'nu20': nu20, 'nu02': nu02, 'nu11': nu11,
        'hu': hu, 'hu_log': hu_log
    }

# ─────────────────────────────────────────────
# 4. CHAIN CODES
# ─────────────────────────────────────────────
def get_chain_code(contour, mode=8):
    pts = contour.reshape(-1, 2)
    if mode == 4:
        dirs = {(1,0):0,(0,1):1,(-1,0):2,(0,-1):3}
    else:
        dirs = {(1,0):0,(1,1):1,(0,1):2,(-1,1):3,
                (-1,0):4,(-1,-1):5,(0,-1):6,(1,-1):7}
    codes = []
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i+1)%len(pts)]
        dx = int(np.sign(p2[0]-p1[0]))
        dy = int(np.sign(p2[1]-p1[1]))
        if (dx,dy) in dirs:
            codes.append(dirs[(dx,dy)])
    return codes

def normalize_chain_code(codes):
    """Normalize for rotation invariance using first difference"""
    n = len(codes) if codes else 0
    if n < 2:
        return codes, codes
    diff = [(codes[i]-codes[i-1])%8 for i in range(1,n)] + [(codes[0]-codes[-1])%8]
    # Normalize start: find lexicographically smallest rotation
    min_rot = min(diff[i:]+diff[:i] for i in range(len(diff)))
    return codes, min_rot

# ─────────────────────────────────────────────
# 5. FOURIER DESCRIPTORS
# ─────────────────────────────────────────────
def get_fourier_descriptors(contour, n_descriptors=None):
    pts = contour.reshape(-1, 2).astype(float)
    z = pts[:,0] + 1j*pts[:,1]
    F = np.fft.fft(z)
    if n_descriptors:
        F_trunc = np.zeros_like(F)
        F_trunc[:n_descriptors//2] = F[:n_descriptors//2]
        F_trunc[-(n_descriptors//2):] = F[-(n_descriptors//2):]
    else:
        F_trunc = F.copy()
    # Normalize: translation (zero DC), scale (|F[1]|=1), rotation (angle F[1]=0)
    F_norm = F_trunc.copy()
    F_norm[0] = 0
    if abs(F_norm[1]) > 0:
        F_norm = F_norm / abs(F_norm[1])
    descriptors = np.abs(F_norm[1:len(F_norm)//2+1])
    return F, F_trunc, F_norm, descriptors

def reconstruct_from_fourier(F_trunc, n_pts=None):
    if n_pts is None:
        n_pts = len(F_trunc)
    z_rec = np.fft.ifft(F_trunc)
    return np.stack([z_rec.real, z_rec.imag], axis=1)

# ─────────────────────────────────────────────
# 6. POLYGONAL APPROXIMATION (Douglas-Peucker)
# ─────────────────────────────────────────────
def douglas_peucker(contour, epsilon_ratio=0.02):
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

# ─────────────────────────────────────────────
# 7. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_all_features(mask):
    props = extract_region_properties(mask)
    moms = compute_moments(mask)
    if 'contour' not in props or props['contour'] is None:
        return None
    cnt = props['contour']
    cc8, cc8_norm = normalize_chain_code(get_chain_code(cnt, 8))
    _, _, _, fd = get_fourier_descriptors(cnt, n_descriptors=20)
    fd_feat = fd[:10] if len(fd) >= 10 else np.pad(fd, (0, 10-len(fd)))
    
    region_feats = [
        props['area']/(128*128),
        props['circularity'],
        props['aspect_ratio'],
        props['extent'],
        props['solidity'],
        props['equiv_diameter']/128,
        props['perimeter']/512,
        props['hull_area']/(128*128)
    ]
    moment_feats = list(moms['hu_log'])
    fourier_feats = list(fd_feat)
    
    return {
        'region': region_feats,
        'moments': moment_feats,
        'fourier': fourier_feats,
        'all': region_feats + moment_feats + fourier_feats,
        'props': props,
        'moms': moms,
        'contour': cnt,
        'chain_code': cc8,
        'chain_norm': cc8_norm
    }

# ─────────────────────────────────────────────
# 8. CLASSIFICATION
# ─────────────────────────────────────────────
def classify(dataset):
    all_feats = {'region':[], 'moments':[], 'fourier':[], 'all':[]}
    labels = []
    raw_data = []
    
    for cls_idx, (cls_name, samples) in enumerate(dataset.items()):
        for mask in samples:
            feats = extract_all_features(mask)
            if feats:
                for k in all_feats:
                    all_feats[k].append(feats[k])
                labels.append(cls_idx)
                raw_data.append({'class': cls_name, 'feats': feats})
    
    label_names = list(dataset.keys())
    results = {}
    
    for feat_name, feat_list in all_feats.items():
        X = np.array(feat_list)
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
        
        knn.fit(X_scaled, y)
        y_pred = knn.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)
        
        results[feat_name] = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'train_acc': accuracy_score(y, y_pred),
            'confusion_matrix': cm,
            'label_names': label_names
        }
    
    return results, raw_data, all_feats, labels

# ─────────────────────────────────────────────
# 9. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_dataset(dataset, save_path):
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    fig.patch.set_facecolor('#0f172a')
    colors = ['#38bdf8', '#34d399', '#f97316']
    for r, (cls_name, samples) in enumerate(dataset.items()):
        for c, mask in enumerate(samples[:10]):
            ax = axes[r, c]
            ax.imshow(mask, cmap='gray', vmin=0, vmax=255)
            ax.set_facecolor('#1e293b')
            if c == 0:
                ax.set_ylabel(cls_name.upper(), color=colors[r], fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[r]); spine.set_linewidth(1.5)
    plt.suptitle('Dataset: 3 Kelas Objek (10 Sampel per Kelas)',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_contours(dataset, save_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor('#0f172a')
    colors = ['#38bdf8', '#34d399', '#f97316']
    
    for r, (cls_name, samples) in enumerate(dataset.items()):
        for c_idx in range(min(4, len(samples))):
            mask = samples[c_idx]
            ax = axes[r, c_idx]
            ax.set_facecolor('#1e293b')
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                pts = cnt.reshape(-1, 2)
                ax.fill(pts[:,0], 128-pts[:,1], color=colors[r], alpha=0.3)
                ax.plot(np.append(pts[:,0], pts[0,0]),
                        128-np.append(pts[:,1], pts[0,1]),
                        color=colors[r], linewidth=2)
                
                # Convex hull
                hull = cv2.convexHull(cnt)
                hp = hull.reshape(-1,2)
                ax.plot(np.append(hp[:,0], hp[0,0]),
                        128-np.append(hp[:,1], hp[0,1]),
                        '--', color='white', linewidth=1, alpha=0.5)
                
                # Bounding box
                x,y,w,h = cv2.boundingRect(cnt)
                rect = patches.Rectangle((x, 128-y-h), w, h,
                    linewidth=1.5, edgecolor='yellow', facecolor='none', linestyle=':')
                ax.add_patch(rect)
                
                # Centroid
                M = cv2.moments(cnt)
                if M['m00']:
                    cxm = M['m10']/M['m00']
                    cym = 128 - M['m01']/M['m00']
                    ax.plot(cxm, cym, '+', color='red', markersize=10, markeredgewidth=2)
                
                # Douglas-Peucker
                approx = douglas_peucker(cnt)
                ap = approx.reshape(-1,2)
                ax.plot(np.append(ap[:,0], ap[0,0]),
                        128-np.append(ap[:,1], ap[0,1]),
                        color='cyan', linewidth=1.5, alpha=0.8)
            
            ax.set_xlim(0,128); ax.set_ylim(0,128)
            ax.set_xticks([]); ax.set_yticks([])
            if c_idx == 0:
                ax.set_ylabel(cls_name.upper(), color=colors[r], fontsize=11, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[r]); spine.set_linewidth(1.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color='#38bdf8', linewidth=2, label='Contour'),
        Line2D([0],[0], color='white', linestyle='--', linewidth=1, alpha=0.7, label='Convex Hull'),
        Line2D([0],[0], color='yellow', linestyle=':', linewidth=1.5, label='Bounding Box'),
        Line2D([0],[0], color='cyan', linewidth=1.5, label='Douglas-Peucker'),
        Line2D([0],[0], marker='+', color='red', markersize=10, linewidth=0, label='Centroid'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5,
               facecolor='#1e293b', edgecolor='gray', labelcolor='white',
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Contour, Convex Hull, Bounding Box & Polygonal Approximation',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_fourier_reconstruction(dataset, save_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor('#0f172a')
    colors = ['#38bdf8', '#34d399', '#f97316']
    n_descs = [5, 10, 20, 'original']
    
    for r, (cls_name, samples) in enumerate(dataset.items()):
        mask = samples[0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        F, _, _, _ = get_fourier_descriptors(cnt)
        
        for c, nd in enumerate(n_descs):
            ax = axes[r, c]
            ax.set_facecolor('#1e293b')
            
            if nd == 'original':
                pts = cnt.reshape(-1,2)
                ax.fill(pts[:,0], 128-pts[:,1], color=colors[r], alpha=0.2)
                ax.plot(np.append(pts[:,0],pts[0,0]),
                        128-np.append(pts[:,1],pts[0,1]),
                        color=colors[r], linewidth=2)
                ax.set_title(f'Original ({len(cnt)} pts)', color='white', fontsize=10)
            else:
                F_tr = np.zeros_like(F)
                half = nd//2
                F_tr[:half] = F[:half]
                F_tr[-half:] = F[-half:]
                rec = reconstruct_from_fourier(F_tr)
                ax.fill(rec[:,0], 128-rec[:,1], color=colors[r], alpha=0.2)
                ax.plot(np.append(rec[:,0],rec[0,0]),
                        128-np.append(rec[:,1],rec[0,1]),
                        color=colors[r], linewidth=2)
                ax.set_title(f'N={nd} descriptors', color='white', fontsize=10)
            
            ax.set_xlim(0,128); ax.set_ylim(0,128)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cls_name.upper(), color=colors[r], fontsize=11, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[r]); spine.set_linewidth(1.5)
    
    plt.suptitle('Rekonstruksi Fourier Descriptors (N=5, 10, 20 vs Original)',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_confusion_matrices(results, save_path):
    feat_names = list(results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0f172a')
    
    for ax, fname in zip(axes, feat_names):
        cm = results[fname]['confusion_matrix']
        labels = results[fname]['label_names']
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
        ax.set_facecolor('#1e293b')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, color='white', fontsize=10)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color='white', fontsize=10)
        ax.set_xlabel('Predicted', color='white', fontsize=10)
        ax.set_ylabel('Actual', color='white', fontsize=10)
        title_map = {'region':'Region Props','moments':'Hu Moments','fourier':'Fourier Desc','all':'All Features'}
        acc = results[fname]['cv_mean']
        ax.set_title(f"{title_map.get(fname,fname)}\nCV Acc: {acc:.1%}", color='white', fontsize=11, fontweight='bold')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]),
                        ha='center', va='center',
                        color='white' if cm[i,j] < cm.max()/2 else 'black',
                        fontsize=14, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
    
    plt.suptitle('Matriks Konfusi k-NN (k=3) per Kategori Fitur',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_accuracy_comparison(results, save_path):
    feat_names = list(results.keys())
    labels_disp = ['Region\nProperties', 'Hu\nMoments', 'Fourier\nDescriptors', 'All\nFeatures']
    cv_means = [results[f]['cv_mean'] for f in feat_names]
    cv_stds  = [results[f]['cv_std']  for f in feat_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    colors_bar = ['#38bdf8','#34d399','#f97316','#a78bfa']
    bars = ax.bar(labels_disp, cv_means, yerr=cv_stds, capsize=8,
                  color=colors_bar, alpha=0.85, error_kw={'color':'white','linewidth':2})
    
    for bar, mean in zip(bars, cv_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{mean:.1%}', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Cross-Validation Accuracy (5-fold)', color='white', fontsize=11)
    ax.set_title('Perbandingan Akurasi Klasifikasi k-NN per Kategori Fitur',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('white')
    ax.set_facecolor('#1e293b')
    ax.grid(axis='y', color='#334155', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

def plot_hu_moments(raw_data, save_path):
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.patch.set_facecolor('#0f172a')
    cls_colors = {'circle':'#38bdf8', 'rectangle':'#34d399', 'triangle':'#f97316'}
    cls_markers = {'circle':'o', 'rectangle':'s', 'triangle':'^'}
    
    for hu_idx, ax in enumerate(axes):
        ax.set_facecolor('#1e293b')
        for cls_name, clr in cls_colors.items():
            vals = [d['feats']['moms']['hu_log'][hu_idx]
                    for d in raw_data if d['class'] == cls_name]
            x_pos = {'circle':0,'rectangle':1,'triangle':2}[cls_name]
            ax.scatter([x_pos]*len(vals) + np.random.uniform(-0.15,0.15,len(vals)),
                       vals, color=clr, alpha=0.7, s=60, marker=cls_markers[cls_name])
            ax.plot([x_pos-0.2, x_pos+0.2], [np.mean(vals)]*2, color='white', linewidth=2.5)
        
        ax.set_xticks([0,1,2]); ax.set_xticklabels(['C','R','T'], color='white', fontsize=10)
        ax.set_title(f'H{hu_idx+1}', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#334155')
        ax.grid(axis='y', color='#334155', linewidth=0.5)
    
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#38bdf8', markersize=8, label='Circle', linewidth=0),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#34d399', markersize=8, label='Rectangle', linewidth=0),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#f97316', markersize=8, label='Triangle', linewidth=0),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               facecolor='#1e293b', edgecolor='gray', labelcolor='white', fontsize=10,
               bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle('Distribusi 7 Hu Invariant Moments per Kelas (log scale)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Generating dataset...")
    dataset = generate_dataset()
    
    print("Extracting features...")
    results, raw_data, all_feats, labels = classify(dataset)
    
    print("Generating visualizations...")
    plot_dataset(dataset, '/home/claude/output/fig1_dataset.png')
    plot_contours(dataset, '/home/claude/output/fig2_contours.png')
    plot_fourier_reconstruction(dataset, '/home/claude/output/fig3_fourier.png')
    plot_confusion_matrices(results, '/home/claude/output/fig4_confusion.png')
    plot_accuracy_comparison(results, '/home/claude/output/fig5_accuracy.png')
    plot_hu_moments(raw_data, '/home/claude/output/fig6_hu_moments.png')
    
    # Save region properties table data
    table_data = []
    for d in raw_data[:9]:  # 3 per class
        p = d['feats']['props']
        m = d['feats']['moms']
        table_data.append({
            'class': d['class'],
            'area': round(p['area'],1),
            'perimeter': round(p['perimeter'],1),
            'cx': round(p['centroid_x'],1),
            'cy': round(p['centroid_y'],1),
            'aspect_ratio': round(p['aspect_ratio'],3),
            'extent': round(p['extent'],3),
            'solidity': round(p['solidity'],3),
            'circularity': round(p['circularity'],3),
            'hu1': round(m['hu_log'][0],4),
            'hu2': round(m['hu_log'][1],4),
            'hu3': round(m['hu_log'][2],4),
        })
    
    # Chain code sample
    chain_samples = {}
    for cls_name, samples in dataset.items():
        mask = samples[0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cc4 = get_chain_code(cnt, 4)
            cc8 = get_chain_code(cnt, 8)
            _, cc8_norm = normalize_chain_code(cc8)
            chain_samples[cls_name] = {
                'cc4_sample': cc4[:20],
                'cc8_sample': cc8[:20],
                'cc8_norm_sample': cc8_norm[:20],
                'length': len(cc8)
            }
    
    output = {
        'results': {k: {
            'cv_mean': v['cv_mean'],
            'cv_std': v['cv_std'],
            'train_acc': v['train_acc'],
            'confusion_matrix': v['confusion_matrix'].tolist()
        } for k, v in results.items()},
        'table_data': table_data,
        'chain_samples': chain_samples
    }
    
    with open('/home/claude/output/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n=== CLASSIFICATION RESULTS ===")
    for feat_name, res in results.items():
        print(f"{feat_name:12s}: CV Acc = {res['cv_mean']:.3f} ± {res['cv_std']:.3f}  |  Train Acc = {res['train_acc']:.3f}")
    
    print("\nAll outputs saved to /home/claude/output/")
    return output

if __name__ == '__main__':
    output = main()
