import os
import time
from loadmodel import sam2_model
from utils import show_mask, show_points, show_box
import numpy as np

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from notebooks.test_opencv import annotate_image

#get frames
#===========================================================================
# Define the video directory
video_dir = "./videos/bedroom"

# Scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Take a look at the first video frame
frame_idx = 0
frame_path = os.path.join(video_dir, frame_names[frame_idx])
print(frame_path)
# Read and display the image using OpenCV
# image = cv2.imread(frame_path)
# cv2.imshow(f"Frame {frame_idx}", image)

# #***
# # Check if image is loaded
# if image is None:
#     print(f"Error: Unable to load image at {video_dir}")
#     exit(1)

# print("showing image")
# cv2.imshow("Bounding Box Annotation", image)
# print('checking mouse click')
# cv2.setMouseCallback("Bounding Box Annotation", draw_bbox)
# print('stoping mouse click check')
# # Start the class selection loop
# change_class()
# print('class selected')
# cv2.destroyAllWindows()
# print('windows closed')

# # Wait for a key press and then close the window
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # wait for 10 sec
# # time.sleep(10)

# # Print all annotated bounding boxes
# print("Annotated Bounding Boxes:", bboxes)
# print("boxes annotated")
# #***

# Call the function to annotate
bboxes = annotate_image(frame_path)

# Print or use the bounding boxes
print("Final Annotated Bounding Boxes:", bboxes)

#initialize inference state
#===========================================================================================

inference_state = sam2_model.init_state(video_path=video_dir)


#if needed..
# sam2_model.reset_state(inference_state)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[210, 350]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits = sam2_model.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# # Save the plot to the directory
# save_path = os.path.join("tempp", f"{frame_idx}_plot_0_.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
# plt.close()  # Close the plot to free memory

# print(f"Plot saved at: {save_path}")
# # # wait for 10 sec
# time.sleep(10)
points=[]
labels=[]
for index, cord in enumerate(bboxes):
    if cord[4] ==1:
        ann_obj_id=1
        # points = np.array([[cord[0], cord[1]]], dtype=np.float32)
        points.append([cord[0], cord[1]])
        if index==0:
            # labels = np.array([1], np.int32)
            labels.append(1)
        else:
            labels.append(0)

points = np.array(points, dtype=np.float32)
labels = np.array(labels, np.int32)

ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
# points = np.array([[210, 350], [250, 220], [275, 175]], dtype=np.float32)

# for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 1, 0], np.int32)
_, out_obj_ids, out_mask_logits = sam2_model.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# Save the plot to the directory
save_path = os.path.join("tempp", f"{frame_idx}_plot_0_.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
plt.close()  # Close the plot to free memory

print(f"Plot saved at: {save_path}")

#object 1 points added(above)....positive and negative
#==========================================================================================================

print('mask shape - ',out_mask_logits.shape)

print('out_obj_ids: ',out_obj_ids)
print('out_mask_logits:',out_mask_logits)

points=[]
labels=[]

for index, cord in enumerate(bboxes):
    if cord[4] ==2:
        ann_obj_id=2
        # points = np.array([[cord[0], cord[1]]], dtype=np.float32)
        points.append([cord[0], cord[1]])
        if index==2:
            # labels = np.array([1], np.int32)
            labels.append(1)
        else:
            labels.append(0)

points = np.array(points, dtype=np.float32)
labels = np.array(labels, np.int32)


ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
# points = np.array([[400, 150]], dtype=np.float32)   
# for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = sam2_model.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)


# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# Save the plot to the directory
save_path = os.path.join("tempp", f"{frame_idx}_plot_1_.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
plt.close()  # Close the plot to free memory

print(f"Plot saved at: {save_path}")

#got masks for object 2(above) ...by adding a point 
#====================================================================================

print('[2-obj] mask shape - ',out_mask_logits.shape)

print('[2-obj] out_obj_ids: ',out_obj_ids)
print('[2-obj] out_mask_logits:',out_mask_logits)

mask_to_pass = out_mask_logits[1]
print('mask shape to pass: ', mask_to_pass[0].shape)

print('stop')

ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([300, 0, 500, 400], dtype=np.float32)
print("box dim: ", box.ndim)
print("box shape: ", box.shape)

# _, out_obj_ids, out_mask_logits = sam2_model.add_new_mask(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     mask=mask_to_pass[0],
# )

# passed the masks to the model for segmentation....(above) 
#=================================================================================================

print('[2-obj mask added] mask shape - ',out_mask_logits.shape)

print('[2-obj mask added] out_obj_ids: ',out_obj_ids)


# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[1])

# Save the plot to the directory
save_path = os.path.join("tempp", f"{frame_idx}_plot_2_.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
plt.close()  # Close the plot to free memory

print(f"Plot saved at: {save_path}")


# video propagation started...(below)
#=================================================================================================================

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# Define the directory to save frames
save_dir = "check_frames"
os.makedirs(save_dir, exist_ok=True)

vis_frame_stride = 2
plt.close("all")

for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"frame {out_frame_idx}")
    ax.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, ax, obj_id=out_obj_id)

    # Save the frame
    save_path = os.path.join(save_dir, f"frame_{out_frame_idx:04d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


cv2.destroyAllWindows()
