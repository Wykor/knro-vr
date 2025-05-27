import cv2
import numpy as np
import time
import tkinter as tk
import os
import glob
from datetime import datetime

class GreenScreenVR:
    """
    Simple modular green-screen camera pipeline for a VR session.

    Keys:
        g - toggle green screen replacement
        f - toggle full-screen video display
        b - switch background
        c - take photo
        q - quit
    """

    def __init__(self, background_path: str = None, camera_index: int = 0):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        # Load background images
        self.background_paths = self._get_background_paths(background_path)
        self.current_bg_index = 0
        self.background = self._load_current_background()
        
        # Photo capture setup
        self.captured_photo = None
        self.session_folder = self._create_session_folder()
        self.photo_count = 0
        self._load_last_photo_from_session()

        # Toggles
        self.use_green_screen = True
        self.full_screen = False

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

        # Get screen resolution for fullscreen resizing
        root = tk.Tk()
        root.withdraw()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()

        # Window setup
        self.win_name = "Video"
        self.info_win = "Info"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(self.info_win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Set window positions - info window on side screen
        cv2.moveWindow(self.win_name, 100, 100)
        # Position info window on main screen first, then you can drag it
        cv2.moveWindow(self.info_win, 800, 100)
        
    def _get_background_paths(self, initial_path):
        """Get list of background image paths."""
        if initial_path and os.path.exists(initial_path):
            backgrounds = [initial_path]
        else:
            backgrounds = []
            
        # Add backgrounds from folder
        bg_folder = "backgrounds"
        if os.path.exists(bg_folder):
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            for pattern in patterns:
                backgrounds.extend(glob.glob(os.path.join(bg_folder, pattern)))
                
        # Fallback to default background
        if not backgrounds and os.path.exists("background.jpg"):
            backgrounds = ["background.jpg"]
            
        if not backgrounds:
            raise RuntimeError("No background images found")
            
        return sorted(backgrounds)
    
    def _load_current_background(self):
        """Load the currently selected background."""
        bg_path = self.background_paths[self.current_bg_index]
        bg = cv2.imread(bg_path)
        if bg is None:
            raise RuntimeError(f"Cannot load background: {bg_path}")
        return bg
        
    def _create_session_folder(self):
        """Create a unique session folder for this run."""
        base_folder = "photos"
        os.makedirs(base_folder, exist_ok=True)
        
        # Create session folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(base_folder, f"session_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)
        
        return session_folder
        
    def _load_last_photo_from_session(self):
        """Load the last captured photo from current session if any exist."""
        if os.path.exists(self.session_folder):
            photo_files = glob.glob(os.path.join(self.session_folder, "photo_*.jpg"))
            if photo_files:
                # Sort by filename to get the latest
                latest_photo = sorted(photo_files)[-1]
                self.captured_photo = cv2.imread(latest_photo)
                # Extract photo number from filename
                filename = os.path.basename(latest_photo)
                try:
                    self.photo_count = int(filename.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    self.photo_count = len(photo_files)
            
    def switch_background(self):
        """Switch to the next background image."""
        self.current_bg_index = (self.current_bg_index + 1) % len(self.background_paths)
        self.background = self._load_current_background()
        
    def take_photo(self, frame):
        """Capture and save the current frame with timestamp."""
        self.photo_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{self.photo_count:03d}_{timestamp}.jpg"
        photo_path = os.path.join(self.session_folder, filename)
        
        cv2.imwrite(photo_path, frame)
        self.captured_photo = frame.copy()
        print(f"Photo saved: {photo_path}")

    # ――― Utility methods ―――
    def _calculate_fps(self):
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        self.fps = 1.0 / dt if dt > 0 else 0.0

    def _resize_background(self, frame_shape):
        """Resize cached background to match incoming frame size."""
        return cv2.resize(self.background, (frame_shape[1], frame_shape[0]))

    def _apply_chroma_key(self, frame):
        """Replace green screen with the loaded background image."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 80, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        bg_resized = self._resize_background(frame.shape)
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
        return cv2.add(fg, bg)

    # ――― Public API ―――
    def process_frame(self, frame):
        """Apply chroma-keying when enabled."""
        return self._apply_chroma_key(frame) if self.use_green_screen else frame

    def overlay_info(self, frame):
        """Draw diagnostics and UI hints onto the frame."""
        info_frame = frame.copy()
        current_bg_name = os.path.basename(self.background_paths[self.current_bg_index])
        
        text_lines = [
            f"FPS: {self.fps:.1f}",
            f"Green Screen: {'ON' if self.use_green_screen else 'OFF'}",
            f"Fullscreen: {'ON' if self.full_screen else 'OFF'}",
            f"Background: {current_bg_name} ({self.current_bg_index + 1}/{len(self.background_paths)})",
            f"Photos Captured: {self.photo_count} (Session: {os.path.basename(self.session_folder)})",
            "'g' toggle, 'f' fullscreen, 'b' background, 'c' photo, 'q'/ESC quit",
        ]
        y0, dy = 30, 30
        for i, text in enumerate(text_lines):
            y = y0 + i * dy
            cv2.putText(info_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
        # Display captured photo in corner if it exists
        if self.captured_photo is not None:
            _, w = info_frame.shape[:2]
            photo_size = 150
            photo_resized = cv2.resize(self.captured_photo, (photo_size, photo_size))
            x_offset = w - photo_size - 10
            y_offset = 10
            info_frame[y_offset:y_offset + photo_size, x_offset:x_offset + photo_size] = photo_resized
            cv2.rectangle(info_frame, (x_offset-2, y_offset-2), 
                         (x_offset + photo_size + 2, y_offset + photo_size + 2), (255, 255, 255), 2)
                        
        return info_frame

    def toggle_green_screen(self):
        """Switch chroma-key mode on/off."""
        self.use_green_screen = not self.use_green_screen

    def toggle_full_screen(self):
        """Switch full-screen display for the video window."""
        self.full_screen = not self.full_screen
        # Set window fullscreen property
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.full_screen else cv2.WINDOW_NORMAL)

    def run(self):
        """Main loop: capture, process, display, handle input."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                self._calculate_fps()
                processed = self.process_frame(frame)

                # Resize video to fill screen when fullscreen is on
                if self.full_screen:
                    display_frame = cv2.resize(processed,
                                               (self.screen_width, self.screen_height))
                else:
                    display_frame = processed

                info = self.overlay_info(processed)

                cv2.imshow(self.win_name, display_frame)
                cv2.imshow(self.info_win, info)

                # Capture keys with longer timeout for better responsiveness
                key = cv2.waitKey(50) & 0xFF
                
                # Handle key presses - print for debugging
                if key != 255:  # A key was pressed
                    print(f"Key pressed: {key} ('{chr(key) if 32 <= key <= 126 else '?'}')")
                
                if key == ord('g') or key == ord('G'):
                    self.toggle_green_screen()
                    print("Toggled green screen")
                elif key == ord('f') or key == ord('F'):
                    self.toggle_full_screen()
                    print("Toggled fullscreen")
                elif key == ord('b') or key == ord('B'):
                    self.switch_background()
                    print("Switched background")
                elif key == ord('c') or key == ord('C'):
                    self.take_photo(processed)
                elif key == ord('q') or key == ord('Q') or key == 27:  # ESC key
                    print("Quitting...")
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    GreenScreenVR().run()