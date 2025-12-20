import cv2
import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as ET
from skimage.measure import regionprops, label
from segment import label_RBC, label_Platelets, segmenting_purple_cells
from preprocess import preprocess_img
from features import extract_filtered_rbc_features

def parse_annotation(xml_path):
    if not os.path.exists(xml_path):
        return {'RBC': [], 'Platelets': []}
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_data = {'RBC': [], 'Platelets': []}
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if 'RBC' in name:
            lbl = 'RBC'
        elif 'Platelets' in name or 'Platelet' in name:
            lbl = 'Platelets'
        else:
            continue
            
        bndbox = obj.find('bndbox')
        # [y0, x0, y1, x1]
        box = (int(bndbox.find('ymin').text), int(bndbox.find('xmin').text), 
               int(bndbox.find('ymax').text), int(bndbox.find('xmax').text))
        gt_data[lbl].append(box)
        
    return gt_data

def calculate_iou(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_count_accuracy(pred_count, gt_count):
    if gt_count == 0:
        return 0.0 if pred_count > 0 else 1.0
    
    error = abs(pred_count - gt_count)
    accuracy = max(0, 1 - (error / gt_count))
    return accuracy

def get_spatial_metrics(detected_props, gt_boxes, iou_threshold=0.5):
    detected_boxes = [prop.bbox for prop in detected_props]
    matched_gt_indices = set()
    tp = 0
    
    for d_box in detected_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt_indices: continue 
            iou = calculate_iou(d_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
            
    fp = len(detected_boxes) - tp
    fn = len(gt_boxes) - tp
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Jaccard Index (Spatial Accuracy)
    spatial_acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return {
        'count': len(detected_boxes),
        'precision': precision,
        'recall': recall,
        'spatial_acc': spatial_acc,
        'boxes': detected_boxes # List of positions
    }

def visualize_comparison(img, filename, gt_boxes, raw_boxes, filt_boxes, cell_type, output_dir):
    COLOR_GT = (0, 255, 0)
    COLOR_RAW = (0, 0, 255)
    COLOR_FILT = (255, 0, 0)

    img_raw = img.copy()
    for (y0, x0, y1, x1) in gt_boxes:
        cv2.rectangle(img_raw, (x0, y0), (x1, y1), COLOR_GT, 2)
    for (y0, x0, y1, x1) in raw_boxes:
        cv2.rectangle(img_raw, (x0, y0), (x1, y1), COLOR_RAW, 1)
    cv2.putText(img_raw, f"RAW {cell_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    img_filt = img.copy()
    for (y0, x0, y1, x1) in gt_boxes:
        cv2.rectangle(img_filt, (x0, y0), (x1, y1), COLOR_GT, 2)
    for (y0, x0, y1, x1) in filt_boxes:
        cv2.rectangle(img_filt, (x0, y0), (x1, y1), COLOR_FILT, 2)
    cv2.putText(img_filt, f"FILTERED {cell_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack((img_raw, img_filt))
    save_path = os.path.join(output_dir, f"{cell_type}_{filename}")
    cv2.imwrite(save_path, combined)

def process_batch(image_dir, annot_dir, viz_dir=None):
    results = []
    files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if viz_dir and not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    print(f"Processing {len(files)} images...")
    print("="*60)

    for idx, filename in enumerate(files):
        img_path = os.path.join(image_dir, filename)
        xml_path = os.path.join(annot_dir, filename.replace('.jpg', '.xml'))
        img = cv2.imread(img_path)
        if img is None: continue

        gt_data = parse_annotation(xml_path)
        gt_rbc_pos = gt_data['RBC']
        gt_plat_pos = gt_data['Platelets']
        gt_rbc_count = len(gt_rbc_pos)
        gt_plat_count = len(gt_plat_pos)

        _, rbc_labels_raw = label_RBC(img)
        rbc_props_raw = regionprops(rbc_labels_raw)
        rbc_raw_metrics = get_spatial_metrics(rbc_props_raw, gt_rbc_pos)

        df_rbc, _ = extract_filtered_rbc_features(img, rbc_labels_raw, min_area=1500, max_area=10000)
        valid_rbc_ids = set(df_rbc['label_id'].values) if not df_rbc.empty else set()
        rbc_props_filtered = [p for p in rbc_props_raw if p.label in valid_rbc_ids]
        rbc_filt_metrics = get_spatial_metrics(rbc_props_filtered, gt_rbc_pos)

        rbc_raw_cnt_acc = calculate_count_accuracy(rbc_raw_metrics['count'], gt_rbc_count)
        rbc_filt_cnt_acc = calculate_count_accuracy(rbc_filt_metrics['count'], gt_rbc_count)

        pre_img = preprocess_img(img, clahe_cliplimit=3, clahe_tileGridSize=(4, 4))
        _, purple_mask = segmenting_purple_cells(pre_img)
        plat_props_raw = regionprops(label(purple_mask))
        plat_raw_metrics = get_spatial_metrics(plat_props_raw, gt_plat_pos)

        _, plat_labels_filtered = label_Platelets(img)
        plat_props_filtered = regionprops(plat_labels_filtered)
        plat_filt_metrics = get_spatial_metrics(plat_props_filtered, gt_plat_pos)

        plat_raw_cnt_acc = calculate_count_accuracy(plat_raw_metrics['count'], gt_plat_count)
        plat_filt_cnt_acc = calculate_count_accuracy(plat_filt_metrics['count'], gt_plat_count)

        results.append({
            'Image': filename,

            'RBC_GT_Count': gt_rbc_count,
            'RBC_GT_Positions': str(gt_rbc_pos),
            
            'RBC_Raw_Count': rbc_raw_metrics['count'],
            'RBC_Raw_Count_Acc': round(rbc_raw_cnt_acc, 3),
            'RBC_Raw_Spatial_Acc': round(rbc_raw_metrics['spatial_acc'], 3),
            'RBC_Raw_Prec': round(rbc_raw_metrics['precision'], 3),
            'RBC_Raw_Positions': str(rbc_raw_metrics['boxes']),

            'RBC_Filt_Count': rbc_filt_metrics['count'],
            'RBC_Filt_Count_Acc': round(rbc_filt_cnt_acc, 3),
            'RBC_Filt_Spatial_Acc': round(rbc_filt_metrics['spatial_acc'], 3),
            'RBC_Filt_Prec': round(rbc_filt_metrics['precision'], 3),
            'RBC_Filt_Positions': str(rbc_filt_metrics['boxes']),

            'Plat_GT_Count': gt_plat_count,
            'Plat_GT_Positions': str(gt_plat_pos),

            'Plat_Raw_Count': plat_raw_metrics['count'],
            'Plat_Raw_Count_Acc': round(plat_raw_cnt_acc, 3),
            'Plat_Raw_Spatial_Acc': round(plat_raw_metrics['spatial_acc'], 3),
            'Plat_Raw_Prec': round(plat_raw_metrics['precision'], 3),
            'Plat_Raw_Positions': str(plat_raw_metrics['boxes']),

            'Plat_Filt_Count': plat_filt_metrics['count'],
            'Plat_Filt_Count_Acc': round(plat_filt_cnt_acc, 3),
            'Plat_Filt_Spatial_Acc': round(plat_filt_metrics['spatial_acc'], 3),
            'Plat_Filt_Prec': round(plat_filt_metrics['precision'], 3),
            'Plat_Filt_Positions': str(plat_filt_metrics['boxes']),
        })
        if viz_dir and idx < 5:
            visualize_comparison(img, filename, gt_rbc_pos, rbc_raw_metrics['boxes'], rbc_filt_metrics['boxes'], "RBC", viz_dir)
            visualize_comparison(img, filename, gt_plat_pos, plat_raw_metrics['boxes'], plat_filt_metrics['boxes'], "Platelets", viz_dir)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}...")

    return pd.DataFrame(results)

INPUT_DIR = '../data/input/JPEGImages/'
ANNOT_DIR = '../data/input/Annotations/'
VIZ_DIR   = 'evaluation_viz/'

if os.path.exists(INPUT_DIR):
    df = process_batch(INPUT_DIR, ANNOT_DIR, VIZ_DIR)
    df.to_csv('final_evaluation_report.csv', index=False)
    print("\n" + "="*30)
    print("      FINAL SUMMARY")
    print("="*30)
    print(f"RBC Count Accuracy:      {df['RBC_Raw_Count_Acc'].mean():.2%} -> {df['RBC_Filt_Count_Acc'].mean():.2%}")
    print(f"RBC Spatial Accuracy:    {df['RBC_Raw_Spatial_Acc'].mean():.2%} -> {df['RBC_Filt_Spatial_Acc'].mean():.2%}")
    print("-" * 30)
    print(f"Platelet Count Accuracy: {df['Plat_Raw_Count_Acc'].mean():.2%} -> {df['Plat_Filt_Count_Acc'].mean():.2%}")
    print(f"Platelet Spatial Acc:    {df['Plat_Raw_Spatial_Acc'].mean():.2%} -> {df['Plat_Filt_Spatial_Acc'].mean():.2%}")
    print(f"\nReport saved to 'final_evaluation_report.csv'")
else:
    print("Input folder not found.")