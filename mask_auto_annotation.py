import os
import cv2
import supervision as sv

from typing import List
from path_config import HOME
from loadmodel import grounding_dino_model

#function to enhance the label prompt
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def auto_BB_annotate(SOURCE_IMAGE_PATH, CLASSES): 
    #gets the basename of image
    image_name = os.path.basename(SOURCE_IMAGE_PATH) 

    #define a dir for saving bounding boxes
    OUTPUT_DIR = f"{HOME}/bounding_box"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #image path with bounding boxes
    OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")

    #define thresholds for box and text
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # Load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # Detect objects
    # detections = grounding_dino_model.predict_with_caption(
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        # caption=CLASSES[0],
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # Annotate image with detections
    box_annotator = sv.BoxAnnotator()
    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}"
    #     for _, _, confidence, class_id, _
    #     in detections]
    labels = [
    f"{CLASSES[class_id] if class_id is not None and class_id < len(CLASSES) else 'unknown'} {confidence:0.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    #get the annotated frame with bounding box
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # Save the annotated image
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
    print(f"Annotated image saved at {OUTPUT_IMAGE_PATH}")

    return image, detections, labels, OUTPUT_IMAGE_PATH


# EXAMPLE USAGE - 
# image, detections = auto_BB_annotate("video_to_frames/00000.jpeg", ['Kids'])
# image, detections = auto_BB_annotate("data/00001.jpeg", ['Kids'])

