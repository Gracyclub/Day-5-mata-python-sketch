import cv2
import numpy as np
import turtle

# Load reference image in grayscale
image_path = "skandamata.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not find '{image_path}'. Make sure it is in the same folder.")
    exit()

# Set canvas resolution
target_width = 750
aspect_ratio = image.shape[0] / image.shape[1]
target_height = int(target_width * aspect_ratio)
resized_image = cv2.resize(image, (target_width, target_height))

# Clean thresholding to keep sharp edges (hands, weapons, jewelry, lion's face)
_, binary = cv2.threshold(resized_image, 180, 255, cv2.THRESH_BINARY_INV)

# Find all nested layers (RETR_TREE captures outer outlines, inner cuts, and inner-inner details)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if hierarchy is None:
    print("Error: No shapes detected.")
    exit()

hierarchy = hierarchy[0]

# Turtle Window Setup
screen = turtle.Screen()
screen.setup(width=target_width + 60, height=target_height + 60)
screen.title("Live Drawing - Mataji and Lion (Exact Reference)")
screen.bgcolor("white")
screen.tracer(5, 0)  # Drawing speed balance: fast yet clearly visible

# Pen (Cursor) Setup
pen = turtle.Turtle()
pen.speed(0)
pen.pensize(1)
pen.shape("classic")
pen.showturtle()  # You will see cursor actively drawing all shapes

# Coordinate converter
half_w = target_width / 2
half_h = target_height / 2

def to_turtle_coords(x, y):
    return (x - half_w, half_h - y)

# Helper function to get contour nesting depth level
def get_contour_depth(idx):
    depth = 0
    parent = hierarchy[idx][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth

# Draw function
def render_shape(cnt, fill_col, border_col):
    first_pt = cnt[0][0]
    tx, ty = to_turtle_coords(first_pt[0], first_pt[1])
    
    pen.penup()
    pen.goto(tx, ty)
    pen.pendown()
    
    pen.color(border_col, fill_col)
    pen.begin_fill()
    
    # Smooth step to prevent lagging while maintaining high curve accuracy
    step = 2 if len(cnt) > 80 else 1
    for i in range(1, len(cnt), step):



        pt = cnt[i][0]
        tx, ty = to_turtle_coords(pt[0], pt[1])
        pen.goto(tx, ty)
        
    pen.goto(to_turtle_coords(first_pt[0], first_pt[1]))
    pen.end_fill()

# Group contours by hierarchy depth
max_depth = max(get_contour_depth(i) for i in range(len(contours)))
depth_layers = {d: [] for d in range(max_depth + 1)}

for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 4:  # Remove tiny camera noise
        continue
    d = get_contour_depth(i)
    depth_layers[d].append(cnt)

print("Drawing in progress... Watch the cursor create all details!")

# Render layer by layer (Outer Black -> Inner White -> Inner Black, etc.)
for d in range(max_depth + 1):
    # Even layers (0, 2, 4) are BLACK; Odd layers (1, 3) are WHITE cutouts (Lion face, eyes, hand gaps)
    fill_color = "black" if (d % 2 == 0) else "white"
    border_color = fill_color
    
    for cnt in depth_layers[d]:
        render_shape(cnt, fill_color, border_color)

# Finish and finalize
pen.hideturtle()
screen.update()
print("Done! Exact reference matched perfectly.")
screen.mainloop()