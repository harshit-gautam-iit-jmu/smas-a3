import cv2
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# ==========================================
# PART 1: IMAGE TRANSFORMATIONS
# ==========================================

image_path = 'Example.png' # Update to .png if your file is a PNG
if not os.path.exists(image_path):
    print(f"Error: Could not find '{image_path}' in the current directory.")
    exit()

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

cx, cy = w / 2, h / 2

# Transformation Matrices
A1 = np.array([[2, 0], [0, 0.5]])
A2 = np.array([[0, -1], [1, 0]])
A3 = np.array([[1, 1], [0, 1]])
A4 = np.array([[-1, 0], [0, 1]])

# For A5 visualization, using a very small non-zero y-scale
A5_vis = np.array([[1, 0], [0, 0.005]]) 
# True A5 is [[1, 0], [0, 0]]

matrices = [
    ("A1: Scaling", A1),
    ("A2: 90 Degree Rotation", A2),
    ("A3: Horizontal Shear", A3),
    ("A4: Y-axis Reflection", A4),
    ("A5: Projection onto X-axis", A5_vis)
]

def apply_transform(A, img):
    # 1. Translate center to origin AND flip y-axis for standard Cartesian logic
    T1 = np.array([[1, 0, -cx],
                   [0, -1, cy],
                   [0, 0, 1]])
    
    # 2. Embed 2D transform into a 3x3 affine matrix
    A_3x3 = np.eye(3)
    A_3x3[:2, :2] = A
    
    # 3. Translate origin back to center AND unflip y-axis
    T2 = np.array([[1, 0, cx],
                   [0, -1, cy],
                   [0, 0, 1]])
    
    # Composite matrix
    M = T2 @ A_3x3 @ T1
    M_cv = M[:2, :]
    
    return cv2.warpAffine(img, M_cv, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

for i, (title, A) in enumerate(matrices):
    transformed_img = apply_transform(A, img)
    axes[i+1].imshow(transformed_img)
    axes[i+1].set_title(title)
    axes[i+1].axis('off')

plt.tight_layout()
print("Close the image window to generate the PDF...")
plt.show() # The script will pause here until you close the matplotlib window

# ==========================================
# PART 2: PDF GENERATION WITH ANSWERS
# ==========================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'SMAS-1 | Assignment 3: Image Transformations', 0, 1, 'C')
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

answers = [
    ("A1 = [[2,0],[0,0.5]] (Scaling)", 
     "T(e1) = [2, 0]^T, T(e2) = [0, 0.5]^T", 
     "2", 
     "No dimension/information is lost.",
     "The image stretches to twice its original width (first column scales x by 2) and compresses to half its height (second column scales y by 0.5)."),
    
    ("A2 = [[0,-1],[1,0]] (90 Degree Rotation)", 
     "T(e1) = [0, 1]^T, T(e2) = [-1, 0]^T", 
     "2", 
     "No dimension/information is lost.",
     "The x-axis maps to the positive y-axis and the y-axis maps to the negative x-axis, visually rotating the image 90 degrees counter-clockwise."),
    
    ("A3 = [[1,1],[0,1]] (Horizontal Shear)", 
     "T(e1) = [1, 0]^T, T(e2) = [1, 1]^T", 
     "2", 
     "No dimension/information is lost.",
     "The x-axis points remain unchanged (first column), while the y-axis points are pushed to the right proportionally to their height (second column), slanting the image."),
    
    ("A4 = [[-1,0],[0,1]] (Reflection in y-axis)", 
     "T(e1) = [-1, 0]^T, T(e2) = [0, 1]^T", 
     "2", 
     "No dimension/information is lost.",
     "The x-coordinates are mapped to their opposite signs (first column is negative) while y-coordinates remain the same, acting as a mirror across the vertical center."),
    
    ("A5 = [[1,0],[0,0]] (Projection onto x-axis)", 
     "T(e1) = [1, 0]^T, T(e2) = [0, 0]^T", 
     "1", 
     "Dimension/information IS lost. The 2D image collapses into a 1D space.",
     "Because the second column is entirely zeros, every pixel's vertical position is forced to zero, collapsing the image into a single flat horizontal line.")
]

for title, a, c, d, e in answers:
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, title, 0, 1)
    
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, f"(a) {a}")
    pdf.multi_cell(0, 8, f"(c) Rank = {c}")
    pdf.multi_cell(0, 8, f"(d) {d}")
    pdf.multi_cell(0, 8, f"(e) {e}")
    pdf.ln(5)

pdf_filename = 'Assignment_Answers.pdf'
pdf.output(pdf_filename)
print(f"Success! '{pdf_filename}' has been created in your folder.")
