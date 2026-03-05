import base64
import io
from PIL import Image, ImageGrab


class VisionSystem:
    """Handles screenshot capture and image file loading for visual context."""

    def capture_screenshot(self):
        """Take a screenshot and return as base64-encoded PNG string."""
        screenshot = ImageGrab.grab()
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

    def load_image_file(self, path: str) -> str | None:
        """Load an image file and return as base64-encoded PNG string.

        Accepts any PIL-supported format (JPEG, PNG, BMP, WebP, etc.).
        Returns None if the file cannot be read.
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"   [Vision] Could not load image '{path}': {e}")
            return None
