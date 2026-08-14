"""
Camera Handler Module
Manages webcam capture and frame processing
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Handles camera capture and frame management.

    Works with any camera regardless of native resolution.
    Frames are downscaled proportionally if wider than max_width,
    preserving the camera's native aspect ratio and full field of view.
    """

    # Cap processing width so MediaPipe stays fast on any camera.
    # Height is computed from the camera's actual aspect ratio.
    MAX_PROCESSING_WIDTH = 1280

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.width:  int = 0   # set after start()
        self.height: int = 0   # set after start()

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> bool:
        """
        Open the camera at its native resolution.
        width/height are populated after this call.

        Returns:
            True if camera started successfully.
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FPS, 30)

        native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute output size: downscale if too wide, keep aspect ratio
        if native_w > self.MAX_PROCESSING_WIDTH:
            scale = self.MAX_PROCESSING_WIDTH / native_w
            self.width  = self.MAX_PROCESSING_WIDTH
            self.height = int(native_h * scale)
        else:
            self.width  = native_w
            self.height = native_h

        self.is_running = True
        print(
            f"Camera started: native {native_w}x{native_h}"
            f" → processing {self.width}x{self.height}"
        )
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame, mirror it, and downscale proportionally if needed.
        The aspect ratio of whatever camera is attached is always preserved.
        """
        if not self.is_running or self.cap is None:
            return False, None

        success, frame = self.cap.read()
        if not success:
            return False, None

        frame = cv2.flip(frame, 1)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        return True, frame
        
    def stop(self):
        """Stop camera capture and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Camera stopped")
            
    def get_fps(self) -> float:
        """Get current camera FPS"""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
        
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
