import numpy as np
import cv2
from scipy.spatial import distance

def compute_centroid(mask):
    """Compute the centroid of a binary mask."""
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None  # Avoid division by zero
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

def compute_iou(mask1, mask2):
    """Compute IoU (Intersection over Union) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def match_masks(mask_flag_list, mask_list, iou_threshold=0.3):
    matched_pairs = []
    new_masks = set(range(len(mask_list)))  # Assume all masks are new initially
    updated_masks = mask_list.copy()
    
    centroids_flag = [compute_centroid(m) for m in mask_flag_list]
    centroids_mask = [compute_centroid(m) for m in mask_list]
    
    print("MASK_FLAG before processing:")
    for idx, mask in enumerate(mask_flag_list):
        print(f"Mask {idx}:")
        print(mask)
    
    print("\nMASK before processing:")
    for idx, mask in enumerate(mask_list):
        print(f"Mask {idx}:")
        print(mask)
    
    for i, centroid_flag in enumerate(centroids_flag):
        if centroid_flag is None:
            continue
        
        distances = [distance.euclidean(centroid_flag, centroid) if centroid else float('inf') for centroid in centroids_mask]
        sorted_indices = np.argsort(distances)  # Get sorted indices based on distance
        
        for min_idx in sorted_indices:
            if distances[min_idx] == float('inf'):
                continue
            
            iou = compute_iou(mask_flag_list[i], mask_list[min_idx])
            
            print(f"Checking MASK_FLAG {i} against MASK {min_idx}, IOU: {iou}")
            
            if iou >= iou_threshold:
                matched_pairs.append((i, min_idx))
                new_masks.discard(min_idx)  # Remove from new masks
                updated_masks[min_idx] = mask_flag_list[i]  # Update mask with MASK_FLAG
                break  # Stop searching after finding a valid match
    
    final_masks = [updated_masks[i] if i in dict(matched_pairs).values() else mask_list[i] for i in range(len(mask_list)) if i in new_masks or i in dict(matched_pairs).values()]
    
    print("\nMatched Pairs:", matched_pairs)
    print("\nFinal Updated Masks:")
    for idx, mask in enumerate(final_masks):
        status = "Updated from MASK_FLAG" if idx in dict(matched_pairs).values() else "New Object"
        print(f"Mask {idx} ({status}):\n{mask}\n")
    
    return final_masks

# # Example Usage
# MASK_FLAG = [
#     np.array([[False, False, True], [False, True, True], [False, False, False]]),  
#     np.array([[True, False, False], [False, True, False], [True, True, False]]),
#     np.array([[False, True, False], [True, False, True], [False, False, True]])
# ]

# MASK = [
#     np.array([[False, True, False], [True, False, False], [False, False, True]]),
#     np.array([[False, False, False], [True, True, False], [False, False, False]]),
#     np.array([[True, False, False], [False, True, True], [True, False, False]]),
#     np.array([[False, False, True], [True, False, False], [False, True, False]]),
#     np.array([[True, True, False], [False, False, False], [False, True, True]])
# ]

# updated_masks = match_masks(MASK_FLAG, MASK)
