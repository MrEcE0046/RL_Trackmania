import numpy as np
import cv2
import math
from screen_capture import capture_screen  # Använd samma funktion som tidigare

def is_wall(pixel):
    # Enklare definition: mörkare än tröskelvärde
    return np.mean(pixel) < 50  # Testa 50–80

def cast_ray(image, origin, angle_deg, max_distance=150):
    angle_rad = math.radians(angle_deg)
    x0, y0 = origin
    for d in range(max_distance):
        x = int(x0 + d * math.cos(angle_rad))
        y = int(y0 + d * math.sin(angle_rad))
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            return d / max_distance  # utanför bilden
        pixel = image[y, x]
        if is_wall(pixel):
            return d / max_distance  # träffade vägg
    return 1.0  # ingen vägg i max_range

def get_radar_readings(image, origin, angles):
    readings = []
    for angle in angles:
        dist = cast_ray(image, origin, angle)
        readings.append(dist)
    return np.array(readings)

if __name__ == "__main__":
    region = {
        "top": 0,
        "left": 0,
        "width": 800,
        "height": 600
    }

    frame = capture_screen(region)
    origin = (400, 380)  # Justera beroende på bilens position

    radar_angles = [-180, -150, -90, -30, 0]  # vänster till höger i grader
    readings = get_radar_readings(frame, origin, radar_angles)

    print("Radar readings:", readings)

    # Visa radar på bilden
    for i, angle in enumerate(radar_angles):
        angle_rad = math.radians(angle)
        dist = readings[i] * 150
        end_x = int(origin[0] + dist * math.cos(angle_rad))
        end_y = int(origin[1] + dist * math.sin(angle_rad))
        cv2.line(frame, origin, (end_x, end_y), (0, 255, 0), 2)

    cv2.circle(frame, origin, 5, (0, 0, 255), -1)
    cv2.imshow("Radar view", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
