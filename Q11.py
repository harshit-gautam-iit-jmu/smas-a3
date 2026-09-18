import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Try importing Colab specific library for file upload
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def upload_image():
    """Handles image upload differently depending on Colab vs Local execution."""
    if IN_COLAB:
        print("Please upload an image (JPG/PNG):")
        uploaded = files.upload()
        file_name = next(iter(uploaded))
        return file_name
    else:
        file_name = input("Enter the path to your image (e.g., 'Example.jpg'): ")
        if not os.path.exists(file_name):
            print("File not found! Exiting.")
            exit()
        return file_name

def show_image(img, title="Image"):
    """Helper to display the image inline."""
    from IPython.display import clear_output
    if IN_COLAB:
        clear_output(wait=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_matrix(A, img, original_shape):
    """Applies a 2x2 transformation matrix about the center of the image."""
    h, w = original_shape[:2]
    cx, cy = w / 2, h / 2

    # 1. Translate center to origin AND flip y-axis for standard math coordinates
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
    
    # Apply transformation. Keeps original canvas size, pads with white.
    return cv2.warpAffine(img, M_cv, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

def main():
    print("=== Image Transformation Toolbox ===")
    img_path = upload_image()
    
    # Load and convert to RGB
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    current_img = original_img.copy()
    show_image(current_img, "Original Image")
    
    while True:
        print("\n--- Menu ---")
        print("1. Rotate")
        print("2. Resize (Scale)")
        print("3. Flip (Reflect)")
        print("4. Shear")
        print("5. Custom Matrix")
        print("6. Reset to Original")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ")
        
        A = np.eye(2)
        transform_name = ""
        
        try:
            if choice == '1':
                angle_deg = float(input("Enter rotation angle in degrees (e.g., 45): "))
                theta = math.radians(angle_deg)
                # Note: negative sin to match standard counter-clockwise rotation
                A = np.array([[math.cos(theta), -math.sin(theta)],
                              [math.sin(theta), math.cos(theta)]])
                transform_name = f"Rotated {angle_deg}°"
                
            elif choice == '2':
                sx = float(input("Enter X scaling factor (e.g., 2 for double width): "))
                sy = float(input("Enter Y scaling factor (e.g., 0.5 for half height): "))
                A = np.array([[sx, 0],
                              [0, sy]])
                transform_name = f"Resized (Sx={sx}, Sy={sy})"
                
            elif choice == '3':
                axis = input("Flip across which axis? (x/y): ").lower()
                if axis == 'x':
                    A = np.array([[1, 0], [0, -1]])
                    transform_name = "Flipped across X-axis"
                elif axis == 'y':
                    A = np.array([[-1, 0], [0, 1]])
                    transform_name = "Flipped across Y-axis"
                else:
                    print("Invalid axis.")
                    continue
                    
            elif choice == '4':
                shx = float(input("Enter horizontal shear factor (e.g., 0.5): "))
                shy = float(input("Enter vertical shear factor (e.g., 0): "))
                A = np.array([[1, shx],
                              [shy, 1]])
                transform_name = f"Sheared (Hx={shx}, Hy={shy})"
                
            elif choice == '5':
                print("Enter the 4 values for the 2x2 matrix [[a, b], [c, d]]:")
                a = float(input("a = "))
                b = float(input("b = "))
                c = float(input("c = "))
                d = float(input("d = "))
                A = np.array([[a, b], [c, d]])
                transform_name = f"Custom Matrix\n[[{a}, {b}],\n [{c}, {d}]]"
                
            elif choice == '6':
                current_img = original_img.copy()
                show_image(current_img, "Reset to Original")
                continue
                
            elif choice == '7':
                print("Exiting Toolbox. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
                continue
            
            # Apply transformation
            current_img = apply_matrix(A, current_img, original_img.shape)
            show_image(current_img, transform_name)
            
        except ValueError:
            print("Invalid input! Please enter numerical values where requested.")

if __name__ == "__main__":
    main()
