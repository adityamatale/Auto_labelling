import os
import cv2
import supervision as sv

from typing import List
from path_config import HOME
from loadmodel import grounding_dino_model
from typing import List, Optional, Tuple
import numpy as np

#function to enhance the label prompt
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


# def auto_BB_annotate(SOURCE_IMAGE_PATH, CLASSES): 
#     #gets the basename of image
#     image_name = os.path.basename(SOURCE_IMAGE_PATH) 

#     #define a dir for saving bounding boxes
#     OUTPUT_DIR = f"{HOME}/bounding_box"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     #image path with bounding boxes
#     OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")

#     #define thresholds for box and text
#     BOX_TRESHOLD = 0.35
#     TEXT_TRESHOLD = 0.25

#     # Load image
#     image = cv2.imread(SOURCE_IMAGE_PATH)

#     # Detect objects
#     # detections = grounding_dino_model.predict_with_caption(
#     detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=enhance_class_name(class_names=CLASSES),
#         # caption=CLASSES[0],
#         box_threshold=BOX_TRESHOLD,
#         text_threshold=TEXT_TRESHOLD
#     )

#     # Annotate image with detections
#     box_annotator = sv.BoxAnnotator()
#     # labels = [
#     #     f"{CLASSES[class_id]} {confidence:0.2f}"
#     #     for _, _, confidence, class_id, _
#     #     in detections]
#     labels = [
#     f"{CLASSES[class_id] if class_id is not None and class_id < len(CLASSES) else 'unknown'} {confidence:0.2f}"
#     for class_id, confidence in zip(detections.class_id, detections.confidence)
#     ]

#     print('labels are:', labels)

#     #get the annotated frame with bounding box
#     annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

#     # Save the annotated image
#     cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
#     print(f"Annotated image saved at {OUTPUT_IMAGE_PATH}")

#     return image, detections, labels, OUTPUT_IMAGE_PATH


# # EXAMPLE USAGE - 
# # image, detections = auto_BB_annotate("video_to_frames/00000.jpeg", ['Kids'])
# # image, detections = auto_BB_annotate("data/00001.jpeg", ['Kids'])


def auto_BB_annotate(SOURCE_IMAGE_PATH: str, CLASSES: List[str]) -> Tuple[np.ndarray, Optional[sv.Detections], List[str], str]:
    try:
        # Basic validation
        if not os.path.exists(SOURCE_IMAGE_PATH):
            raise FileNotFoundError(f"Image file not found: {SOURCE_IMAGE_PATH}")
        
        if not CLASSES:
            raise ValueError("CLASSES list cannot be empty")

        image_name = os.path.basename(SOURCE_IMAGE_PATH)
        OUTPUT_DIR = f"{HOME}/bounding_box"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")

        # Load image
        image = cv2.imread(SOURCE_IMAGE_PATH)
        if image is None:
            raise ValueError(f"Could not read image from {SOURCE_IMAGE_PATH}")

        # Define thresholds
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        CONFIDENCE_THRESHOLD = 0.40

        # Detect objects with error handling
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # Handle case where detections is None or invalid
        if detections is None or not hasattr(detections, 'confidence'):
            print("No valid detections returned from model")
            return image, None, [], OUTPUT_IMAGE_PATH

        # Filter detections by confidence threshold
        valid_indices = [
            i for i, conf in enumerate(detections.confidence) 
            if conf is not None and conf >= CONFIDENCE_THRESHOLD
        ]

        if not valid_indices:
            print("No detections meet the confidence threshold")
            return image, sv.Detections.empty(), [], OUTPUT_IMAGE_PATH

        # Create filtered detections
        filtered_detections = sv.Detections(
            xyxy=detections.xyxy[valid_indices],
            confidence=np.array([detections.confidence[i] for i in valid_indices]),
            class_id=np.array([detections.class_id[i] for i in valid_indices]),
            mask=detections.mask[valid_indices] if hasattr(detections, 'mask') and detections.mask is not None else None,
            tracker_id=detections.tracker_id[valid_indices] if hasattr(detections, 'tracker_id') and detections.tracker_id is not None else None
        )

        # Generate labels with additional safety checks
        labels = []
        for class_id, confidence in zip(filtered_detections.class_id, filtered_detections.confidence):
            if class_id is not None and 0 <= class_id < len(CLASSES):
                labels.append(f"{CLASSES[class_id]} {confidence:0.2f}")
            else:
                labels.append(f"unknown {confidence:0.2f}")

        # Annotate image
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy(),
            detections=filtered_detections,
            labels=labels
        )

        # Save the annotated image
        cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
        print(f"Successfully saved annotated image at {OUTPUT_IMAGE_PATH}")

        return image, filtered_detections, labels, OUTPUT_IMAGE_PATH

    except Exception as e:
        print(f"Error in auto_BB_annotate: {str(e)}")
        # Return empty results in case of error
        return image if 'image' in locals() else None, None, [], OUTPUT_IMAGE_PATH if 'OUTPUT_IMAGE_PATH' in locals() else ""
    
