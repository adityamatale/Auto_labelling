import os
import cv2
import numpy as np
import supervision as sv

from path_config import HOME
from loadmodel import sam_predictor
from segment_anything import SamPredictor

#generate segmentation masks
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def segment_instances(SOURCE_IMAGE_PATH, image, detections, CLASSES):
    #get base image name
    image_name = os.path.basename(SOURCE_IMAGE_PATH) 

    #define output dir for segmentation of detected frames 
    OUTPUT_DIR = f"{HOME}/instance_segmentation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"segmented_{image_name}")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}"
    #     for _, _, confidence, class_id, _
    #     in detections]
    labels = [
    f"{CLASSES[class_id] if class_id is not None and class_id < len(CLASSES) else 'unknown'} {confidence:0.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    #get the annotated frame with segmentation mask
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    #get the annotated frame with bounding box
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)


    # Save the segmented image
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_image)
    print(f"Segmented image saved at {OUTPUT_IMAGE_PATH}")


    return detections.mask