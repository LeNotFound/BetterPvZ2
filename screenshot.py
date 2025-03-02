import subprocess
import io
import time
from PIL import Image

class Screenshot:
    def __init__(self, interval_ms=200):
        self.interval = interval_ms / 1000.0  # 转换为秒
        self.last_capture_time = 0
        self.cached_image = None

    def capture(self) -> Image.Image:
        current_time = time.time()
        if self.cached_image is not None and (current_time - self.last_capture_time) < self.interval:
            return self.cached_image
        proc = subprocess.run(["adb", "exec-out", "screencap", "-p"],
                              stdout=subprocess.PIPE)
        image_bytes = proc.stdout
        self.cached_image = Image.open(io.BytesIO(image_bytes))
        self.last_capture_time = current_time
        return self.cached_image

# 供调试使用
if __name__ == "__main__":
    ss = Screenshot(200)
    img = ss.capture()
    img.show()
