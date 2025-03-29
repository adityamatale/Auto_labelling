import cv2

# Initialize global variables
drawing = False
start_x, start_y = -1, -1
bboxes = []  # Store bounding boxes with class labels
selected_class = 1  # Default class
img = None  # Global image variable

def draw_bbox(event, x, y, flags, param):
    """Handles mouse events for drawing bounding boxes."""
    global start_x, start_y, drawing, bboxes, img, selected_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        print(f"Left button clicked at ({start_x}, {start_y}) with class {selected_class}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.putText(img_copy, f"Class {selected_class}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Bounding Box Annotation", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        bboxes.append((start_x, start_y, end_x, end_y, selected_class))

        # Draw final bounding box
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(img, f"Class {selected_class}", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        print(f"Bounding Box Saved: ({start_x}, {start_y}, {end_x}, {end_y}) | Class {selected_class}")
        cv2.imshow("Bounding Box Annotation", img)

def change_class():
    """Allows the user to change the class label using number keys (1-9)."""
    global selected_class
    while True:
        key = cv2.waitKey(0) & 0xFF
        if ord('1') <= key <= ord('9'):  # Allow classes 1 to 9
            selected_class = key - ord('0')  # Convert ASCII to integer
            print(f"Selected Class: {selected_class}")
        elif key == 27:  # ESC key to exit
            break

def annotate_image(image_path):
    """Loads an image and starts annotation process."""
    global img, bboxes
    bboxes=[]
    img = cv2.imread(image_path)

    # Check if image is loaded
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    cv2.imshow("Bounding Box Annotation", img)
    cv2.setMouseCallback("Bounding Box Annotation", draw_bbox)

    # Start class selection loop
    change_class()

    cv2.destroyAllWindows()

    # Return annotated bounding boxes
    return bboxes
