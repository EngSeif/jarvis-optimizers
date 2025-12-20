import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
from scipy import ndimage
import os
from utils.helpers import show_images
from skimage.measure import regionprops, label
from preprocess import preprocess_img
from skimage.filters import frangi
import pandas as pd
from segment import *

def extract_filtered_rbc_features(img, labels, min_area=1500, max_area=10000, remove_borders=True):
    """
    RBC SPECIFIC: Applies strict size and border filtering.
    """
    properties = regionprops(labels)
    rbc_data = []
    valid_ids = []
    img_h, img_w = img.shape[:2]
    for prop in properties:
        # 1. Border Filtering
        if remove_borders:
            y0, x0, y1, x1 = prop.bbox
            if y0 == 0 or x0 == 0 or y1 == img_h or x1 == img_w:
                continue
        # 2. Size Filtering
        if prop.area < min_area or prop.area > max_area:
            continue
        circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
        valid_ids.append(prop.label)
        filtered_mask = np.where(np.isin(labels, valid_ids), labels, 0).astype(np.uint8)
        rbc_data.append({
            "label_id": prop.label,
            "Area": prop.area,
            "Circularity": circularity,
            "Aspect Ratio": aspect_ratio
        })
    return pd.DataFrame(rbc_data),filtered_mask

def extract_platelet_features(img, labels):
    properties = regionprops(labels)
    plat_data = []
    valid_ids = []
    
    # 1. Initialize mask to zeros (Safe default)
    filtered_mask = np.zeros_like(labels, dtype=np.uint8)

    for prop in properties:
        circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
        
        valid_ids.append(prop.label)
        
        plat_data.append({
            "Label ID": prop.label,
            "Area": prop.area,
            "Circularity": circularity,
            "Aspect Ratio": aspect_ratio
        })
        
    # 2. Create the mask ONCE at the end
    if valid_ids:
        filtered_mask = np.where(np.isin(labels, valid_ids), labels, 0).astype(np.uint8)
        
    return pd.DataFrame(plat_data), filtered_mask

def visualize_filtered_rbcs(img, original_labels, filtered_df):
    viz_img = img.copy()
    if filtered_df.empty:
        return viz_img
    valid_ids = set(filtered_df['label_id'].values)
    for prop in regionprops(original_labels):
        if prop.label in valid_ids:
            y0, x0, y1, x1 = prop.bbox
            cv2.rectangle(viz_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(viz_img, "RBC", (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return viz_img

def visualize_filtered_platelets(img, original_labels, filtered_df):
    viz_img = img.copy()
    if filtered_df.empty:
        return viz_img
    valid_ids = set(filtered_df['Label ID'].values)
    for prop in regionprops(original_labels):
        if prop.label in valid_ids:
            y0, x0, y1, x1 = prop.bbox
            cv2.rectangle(viz_img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(viz_img, "Platelet", (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return viz_img

def compute_rbc_parameters(df, img_shape,rbc_mask):
    """
    Calculates and PRINTS the table specifically for RBCs.
    """
    img_h, img_w = img_shape[:2]
    total_area = img_h * img_w
    count = len(df)
    avg_size = df['Area'].mean() if not df.empty else 0
    avg_circ = df['Circularity'].mean() if not df.empty else 0
    avg_ar = df['Aspect Ratio'].mean() if not df.empty else 0
    density = (count / total_area) * 10000
    # 1. Create a binary mask of all valid cells
    binary_mask = (rbc_mask > 0).astype(np.uint8)
    # 2. Close gaps: Watershed puts a 1px line between cells. We dilate to bridge them back 
    kernel = np.ones((3,3), np.uint8)
    merged_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    # 3. Find connected components (blobs) on this merged mask
    num_blobs, blob_labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask)
    # 4. Identify clumps.
    #    A blob is a "clump" if its area is significantly larger than the average single cell.
    clump_threshold = avg_size * 1.5
    clump_area_sum = 0
    total_cell_area = df['Area'].sum()

    for i in range(1, num_blobs):
        blob_area = stats[i, cv2.CC_STAT_AREA]
        if blob_area > clump_threshold:
            clump_area_sum += blob_area
    #    Result is 0% if all cells are separated, 100% if everything is one giant blob.
    overlap_ratio = (clump_area_sum / total_cell_area) * 100 if total_cell_area > 0 else 0
    print("\n" + "="*50)
    print("      RBC PARAMETER REPORT")
    print("="*50)
    print(f"{'Parameter':<25} | {'Value':<20}")
    print("-" * 50)
    print(f"{'Cell Count':<25} | {count}")
    print(f"{'Average Cell Size':<25} | {round(avg_size, 2)} px")
    print(f"{'Circularity Index':<25} | {round(avg_circ, 3)}")
    print(f"{'Aspect Ratio':<25} | {round(avg_ar, 3)}")
    print(f"{'Cell Density':<25} | {round(density, 2)} / 10k px")
    print(f"{'Overlap Ratio':<25} | {round(overlap_ratio, 2)} %")
    print("="*50 + "\n")

def compute_platelet_parameters(df, img_shape):
    """
    Calculates and PRINTS the table specifically for Platelets.
    """
    img_h, img_w = img_shape[:2]
    total_area = img_h * img_w
    count = len(df)
    avg_size = df['Area'].mean() if not df.empty else 0
    avg_circ = df['Circularity'].mean() if not df.empty else 0
    avg_ar = df['Aspect Ratio'].mean() if not df.empty else 0
    density = (count / total_area) * 10000
    overlap = (df['Area'].sum() / total_area) * 100 if not df.empty else 0
    print("\n" + "="*50)
    print("      PLATELET PARAMETER REPORT")
    print("="*50)
    print(f"{'Parameter':<25} | {'Value':<20}")
    print("-" * 50)
    print(f"{'Cell Count':<25} | {count}")
    print(f"{'Average Cell Size':<25} | {round(avg_size, 2)} px")
    print(f"{'Circularity Index':<25} | {round(avg_circ, 3)}")
    print(f"{'Aspect Ratio':<25} | {round(avg_ar, 3)}")
    print(f"{'Cell Density':<25} | {round(density, 2)} / 10k px")
    print(f"{'Overlap Ratio':<25} | {round(overlap, 2)} %")
    print("="*50 + "\n")


img = cv2.imread('../data/input/JPEGImages/BloodImage_00002.jpg')
rbc_image, rbc_labels_all = label_RBC(img)
raw_rbc = regionprops(rbc_labels_all)
raw_rbc_count = len(raw_rbc)
print(f"{'Red Blood Cell Count':<25} | {raw_rbc_count}")

df_rbc, rbc_filtered_mask = extract_filtered_rbc_features(
    img, 
    rbc_labels_all, 
    min_area=1500, 
    max_area=10000, 
    remove_borders=True
)

# 2. Platelet Processing
_, platelet_labels_all = label_Platelets(img)
df_platelets, platelet_filtered_mask = extract_platelet_features(img, platelet_labels_all)

# 3. Reports
compute_rbc_parameters(df_rbc, img.shape, rbc_filtered_mask)
compute_platelet_parameters(df_platelets, img.shape)

# 4. Pixel-based Visualization (Colored Masks)
final_viz = np.zeros_like(img)
final_viz[rbc_filtered_mask > 0] = [0, 255, 0]       # Green
final_viz[platelet_filtered_mask > 0] = [255, 0, 0]  # Blue/Red (BGR vs RGB depending on lib)
overlay = cv2.addWeighted(img, 0.7, final_viz, 0.3, 0)

show_images(
    [rbc_filtered_mask, platelet_filtered_mask, overlay], 
    ["RBC Mask (Filtered)", "Platelet Mask", "Combined Overlay"]
)

# 5. Box-based Visualization
img_with_rbcs = visualize_filtered_rbcs(img, rbc_labels_all, df_rbc)
final_combined_img = visualize_filtered_platelets(img_with_rbcs, platelet_labels_all, df_platelets)

show_images(
    [img,rbc_image ,final_combined_img], 
    ["Original Image","Before filtering" ,"Detected Cells (Green=RBC, Red=Platelet)"]
)