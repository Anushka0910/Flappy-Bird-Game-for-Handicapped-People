import cv2
import numpy as np

class HeadController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Use the webcam
        self.movements = []  # List to store movement positions for smoothing
        self.max_movements = 5  # Moving average window size
        self.previous_center_y = None  # To track the previous center for movement

    def get_head_movement(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # Flip frame to act as a mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained frontal face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        movement_direction = 'neutral'

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face (for debugging)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get the center of the face for movement detection
            center_x = x + w // 2
            center_y = y + h // 2

            # Store the center_y for smoothing
            self.movements.append(center_y)

            # Keep the movement history to a fixed size
            if len(self.movements) > self.max_movements:
                self.movements.pop(0)

            # Calculate the smoothed average of the movements
            avg_center_y = sum(self.movements) / len(self.movements)

            # Detect movement direction based on smoothed y-axis position
            frame_mid_y = frame.shape[0] // 2
            threshold = 20  # Adjust sensitivity here for up/down

            if avg_center_y < frame_mid_y - threshold:
                movement_direction = 'up'  # Tilt head up
            elif avg_center_y > frame_mid_y + threshold:
                movement_direction = 'down'  # Tilt head down
            else:
                movement_direction = 'neutral'  # No movement

        # Debugging: Show the webcam feed and rectangle
        cv2.imshow('Head Control', frame)

        return movement_direction

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# For testing purposes
if __name__ == "__main__":
    controller = HeadController()
    while True:
        direction = controller.get_head_movement()
        print("Direction: ", direction)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    controller.release()
