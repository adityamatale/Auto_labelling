import os
import cv2
import shutil
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from path_config import HOME
from loadmodel import sam2_model
from utils import show_mask, show_box

# get frames from videos
def video_to_frames(VIDEO_PATH, FRAMES_DIR_PATH):

    # capture the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    # get fps and the number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # initialize a scaling to 50%
    SCALE_FACTOR = 0.5

    # generate frames using supervision
    frames_generator = sv.get_video_frames_generator(VIDEO_PATH)
    # frames_generator = sv.get_video_frames_generator(VIDEO_PATH,stride=10)
    sink = sv.ImageSink(
        target_dir_path=FRAMES_DIR_PATH,
        image_name_pattern="{:05d}.jpeg")

    with sink:
        for frame in frames_generator:
            # scale image
            frame = sv.scale_image(frame, SCALE_FACTOR)
            # save image
            sink.save_image(frame)

    return fps, totalNoFrames


# EXAMPLE USAGE
# VIDEO_PATH=os.path.join(HOME, "video_data", "basketball_1.mp4")
FRAMES_DIR_PATH=os.path.join(HOME, "data")
# fps, totalNoFrames=video_to_frames(VIDEO_PATH, FRAMES_DIR_PATH)


# to add the segment masks to the model
def bbox_to_sam(skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, MASK):

    # define inference frame dir
    FRAMES_DIR_PATH = os.path.join(HOME, "inference_frames")

    # save inference frames in dir (source img to int(fps/4))
    #___________________________________________________________
    # Create destination directory if it doesn't exist
    os.makedirs(FRAMES_DIR_PATH, exist_ok=True)

    # Get sorted list of images (numeric sorting)
    image_list = sorted(
        [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpeg")],
        key=lambda x: int(os.path.splitext(x)[0])  # Extract number and sort numerically
    )

    # Get the index of the input image
    input_image_name = os.path.basename(source_image_path)

    if input_image_name in image_list:
        start_index = image_list.index(input_image_name)  # Get image from input image
        end_index = min(start_index + skip_frames, len(image_list))  # Ensure within bounds

        # Copy the frames
        copied_images = []
        # Copy the next 10 images
        for i in range(start_index, end_index):
            src_path = os.path.join(SOURCE_DIR, image_list[i])
            dest_path = os.path.join(FRAMES_DIR_PATH, image_list[i])
            shutil.copy2(src_path, dest_path)  # Use shutil.move() if needed
            copied_images.append(dest_path)
            print(f"Copied: {image_list[i]}")

        print("Process completed!")
    else:
        print("Input image not found in the directory.")


    # Initialize inference state
    inference_state = sam2_model.init_state(FRAMES_DIR_PATH)
    print("Inference state initialized")

    
    # Remove the copied frames after inference
    for image_path in copied_images:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")

    print("Cleanup completed!")

    boxes_by_label = {label: [] for label in CLASSES}

    # Group bounding boxes by label
    for box, class_id in zip(detections.xyxy, detections.class_id):
        label = CLASSES[class_id]
        boxes_by_label[label].append(box)

    print(f"Boxes grouped by label: {boxes_by_label}")

    object_ids = []
    mask_logits = []
    object_id_counter = 1  # Start unique ID counter from 1

    # Iterate over all labels and their bounding boxes
    for label, boxes in boxes_by_label.items():
        if not boxes:  # Skip labels with no bounding boxes
            continue

        print(f"\nProcessing label: {label}")

        for box in boxes:  # Assign a new object ID for each bounding box
            x, y, w, h = map(int, box)
            print(f"  - Bounding box_before (x, y, w, h): ({x}, {y}, {w}, {h})")

            # Create mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[y:h, x:w] = 255
            
            print(f"  - [{object_id_counter-1}] Mask shape: {MASK[object_id_counter-1].shape}, Nonzero pixels: {np.count_nonzero(MASK)}")
           
            print(f"  - [_] Mask shape: {mask.shape}, Nonzero pixels: {np.count_nonzero(mask)}")

            # Assign a unique object ID to each mask
            _, obj_ids, mask_logit = sam2_model.add_new_mask(
                inference_state=inference_state,
                frame_idx=inference_index,
                obj_id=object_id_counter,  # Unique ID per bounding box
                mask=MASK[object_id_counter-1]
            )

            # show the results on the current (interacted) frame
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {FRAME_IDX}")
            plt.imshow(Image.open(f"{source_image_path}"))
            show_box(box, plt.gca())
            show_mask((mask_logit[object_id_counter-1] > 0.0).cpu().numpy(), plt.gca(), obj_id=obj_ids[object_id_counter-1])
            
            # Save the plot to the directory
            save_path = os.path.join("bbox_output", f"{FRAME_IDX}_plot_{object_id_counter}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
            plt.close()  # Close the plot to free memory

            print(f"Plot saved at: {save_path}")

            print(f"  - Assigned Object ID: {object_id_counter}")
            print(f"  - Object IDs returned: {obj_ids}")
            print(f"  - Mask logit shape: {mask_logit.shape if isinstance(mask_logit, np.ndarray) else type(mask_logit)}")

            # object_ids.extend(obj_ids)  # Collect object IDs
            object_ids=obj_ids
            # mask_logits.append(mask_logit)  # Collect mask logits
            mask_logits=mask_logit

            

            object_id_counter += 1  # Increment object ID for the next bounding box

    print(f"\nFinal Object IDs: {object_ids}")
    print(f"Final number of mask logits: {len(mask_logits)}")

    return object_ids, mask_logits, inference_state


# to pass the masks to the skip frames
def visualize_segment_video(index, SOURCE_VIDEO, inference_state):
    SCALE_FACTOR = 1.0
    FRAMES_DIR_PATH = os.path.join(HOME, "data")
    SOURCE_FRAMES = Path(FRAMES_DIR_PATH)

    # Extract video info
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    video_info.width = int(video_info.width * SCALE_FACTOR)
    video_info.height = int(video_info.height * SCALE_FACTOR)

    # Initialize MaskAnnotator
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS)

    # Define target video path
    TARGET_VIDEO = os.path.join(HOME, "video_data", f"{Path(SOURCE_VIDEO).stem}-result_sink.mp4")
    TARGET_VIDEO = Path(TARGET_VIDEO)
    os.makedirs(TARGET_VIDEO.parent, exist_ok=True)

    # Get sorted list of frame paths
    SOURCE_FRAME_PATHS = sorted(sv.list_files_with_extensions(SOURCE_FRAMES.as_posix(), extensions=["jpeg"]))

    # initialize the propagation index
    propagation_index = 0
    # start the propagation
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
        frame_idx = propagation_index+index
        print(f"Processing frame {frame_idx}, mask_logits shape: {mask_logits.shape}")

        if frame_idx >= len(SOURCE_FRAME_PATHS):
            print(f"⚠️ Frame index {frame_idx} out of range! Skipping...")
            continue

        frame_path = SOURCE_FRAME_PATHS[frame_idx]
        frame = cv2.imread(frame_path)

        if frame is None or frame.size == 0:
            print(f"⚠️ Skipping unreadable frame {frame_idx}!")
            continue

        if frame.shape[2] == 1:  # Convert grayscale to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        masks = (mask_logits > 0.0).cpu().numpy()
        if masks.ndim == 4:
            masks = np.squeeze(masks, axis=1)

        if not np.any(masks):
            print(f"Skipping segmentation for frame {frame_idx}: No segmentation found")
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            cv2.imwrite(f"segmented_frames/{TARGET_VIDEO.stem}_frame_{frame_idx}_original.jpg", frame)
            continue

        # convert masks to BB
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array(object_ids)
        )

        # annotate box on the image
        annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)

        # condition to check empty frames
        if annotated_frame is None or annotated_frame.size == 0:
            print(f"Skipping empty frame {frame_idx}!")
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            cv2.imwrite(f"segmented_frames/{TARGET_VIDEO.stem}_frame_{frame_idx}_original.jpg", frame)
            # cv2.imwrite(f"segmented_frames/frame_{frame_idx}.jpg", frame)

            continue

        print(f"Saving segmented frame {frame_idx}...")
        cv2.imshow("Segmented Frame", annotated_frame)
        cv2.waitKey(1)
        cv2.imwrite(f"segmented_frames/frame_{frame_idx}.jpg", annotated_frame)

        # update propagation index
        propagation_index+=1

    cv2.destroyAllWindows()
    return video_info 
