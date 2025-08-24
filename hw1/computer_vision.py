import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class ObjectDistanceAndPosition:
    def __init__(self, image_path, lower_yellow=None, upper_yellow=None, min_area=100):
        self.image_path = image_path
        self.image_bgr = cv2.imread(image_path)
        if self.image_bgr is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to RGB and HSV
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        
        # HSV thresholds for yellow detections
        self.lower_yellow = lower_yellow if lower_yellow is not None else np.array([20, 100, 100])
        self.upper_yellow = upper_yellow if upper_yellow is not None else np.array([35, 255, 255])
        
        self.min_area = min_area    # Minimum contour area
        self.mask = None            
        self.result_image = None    # Masked image with yellow detection
        self.position_image = None  # Image with bounding boxes and centroids
        
        self.reference_points = []  

    def create_mask(self):
        """ Create and clean mask for yellow objects """
        raw_mask = cv2.inRange(self.image_hsv, self.lower_yellow, self.upper_yellow)    # Pixels in yellow become white (1), else black (0)
        kernel = np.ones((5,5), np.uint8)               
        mask_clean = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)    # Fill small gaps inside objects
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)   # Removes small noise outside objects
        self.mask = mask_clean
        return self.mask

    def apply_mask(self):
        """ Detect contours, filter small ones, and draw bounding boxes with a single centroid """
        if self.mask is None:   
            self.create_mask()  # Ensure mask exists
        self.result_image = cv2.bitwise_and(self.image_rgb, self.image_rgb, mask=self.mask) # Keeps only yellow areas in the RGB image; everything else becomes black
        return self.result_image

    def find_positions(self, box_color=(255,0,0), center_color=(255,0,255), box_thickness=2):
        """Display the image with positions (centroids) and bounding boxes."""
        if self.mask is None:
            self.create_mask() # Ensure mask exists
        
        self.position_image = self.image_rgb.copy() # Create a copy to draw on
        self.centroids = []   # <--- initialize empty list for rotor centroids

        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Find external contours in the mask
        
        for cnt in contours:    
            if cv2.contourArea(cnt) < self.min_area: # Loop through each contour, ignore those too small
                continue 
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            self.centroids.append((cx, cy))   # <--- store each centroid

            cv2.rectangle(self.position_image, (x, y), (x+w, y+h), box_color, box_thickness)
            cv2.circle(self.position_image, (cx, cy), 8, center_color, -1)

        # Draw reference points and horizontal lines
        # Inside your drawing loop
        for px, py in self.reference_points:
            cv2.circle(self.position_image, (px, py), 9, (0, 255, 0), -1)
            line_length = 350
            start_point = (px - line_length//2, py)
            end_point   = (px + line_length//2, py)
            cv2.line(self.position_image, start_point, end_point, (0, 255, 0), 5)

        for (cx, cy) in self.centroids:
            dx, dy = cx - px, cy - py

            # Absolute angle wrt horizontal axis (always 0–90°)
            angle_rad = math.atan2(abs(dy), abs(dx))
            angle_deg = math.degrees(angle_rad)

            # Draw line from reference to rotor
            cv2.line(self.position_image, (px, py), (cx, cy), (255, 0, 255), 2)

            # Display angle slightly above midpoint
            mx, my = (px + cx)//2, (py + cy)//2
            cv2.putText(self.position_image,
                        f"{angle_deg:.1f}°",
                        (mx, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA)



        return self.position_image


        """ FUNCTION TO OBTAIN REFERENCE POINTS
        def select_reference_points(self):
            # Select reference points from images 
        self.reference_points = []
        clone = self.image_rgb.copy()
        cv2.namedWindow("Select Reference Points", cv2.WINDOW_NORMAL)   # Allows to modify window size
        cv2.resizeWindow("Select Reference Points", 800, 600) 
        
        def click_event(event, x, y, flags, param):     # Called everytime the user interacts with the window
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked point: ({x}, {y})")
                self.reference_points.append((x, y))
                cv2.circle(clone, (x, y), 8, (0, 0, 255), -1)
                cv2.imshow("Select Reference Points", clone)    # Update displayed window to show newly drawn circle
        
        # OpenCV calls this function every time the user clicks in the window.
        cv2.setMouseCallback("Select Reference Points", click_event)    
        print("Click on the reference points. Press 'q' to exit.")
        
        while True:
            cv2.imshow("Select Reference Points", clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
        cv2.destroyWindow("Select Reference Points")
        print(f"Reference points updated: {self.reference_points}")
        return self.reference_points
        """

    def show_positions(self, other_image=None, titles=("Image", "Other Image")):
        """Display the image with positions (centroids) and bounding boxes."""
        if self.position_image is None:
            self.find_positions()
        
        if other_image is None: # Display a single image
            plt.imshow(self.position_image)
            plt.title(titles[0])
            plt.axis("off")
            plt.show()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12,6))
            axs[0].imshow(self.position_image)
            axs[0].set_title(titles[0])
            axs[0].axis("off")
            
            axs[1].imshow(other_image)
            axs[1].set_title(titles[1])
            axs[1].axis("off")
            plt.tight_layout()  
            plt.show()