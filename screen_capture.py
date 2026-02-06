import mss
import numpy as np
import cv2

""" 
Kör programmet och kontrollera var på skärmen mss kommer övervaka spelet.
"""

def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if __name__ == "__main__":
    region = {
        "top":  0,
        "left": 0,
        "width": 800,
        "height": 600
    }
    frame = capture_screen(region)
    cv2.imshow("Skärm", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()