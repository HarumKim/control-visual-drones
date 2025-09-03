import cv2, numpy as np, math, time
import socket
import struct

# Configuración UDP
UDP_ADDR_CMD = ('127.0.0.1', 5005)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class LiveAeroTracker:
    def __init__(self, cam_index=0, min_area_ratio=0.0005, alpha=0.3):
        self.cap = cv2.VideoCapture(cam_index)
        # Try to set a reasonable resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initial HSV (tune as needed)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        self.min_area_ratio = min_area_ratio  # min contour area as % of frame
        self.alpha = alpha                    # EMA smoothing factor
        self.ema_ref = None                   # smoothed reference point
        self.ema_centroids = [None, None]     # smoothed rotor centroids
        self.fixed_ref = None


    def _smooth(self, old, new):
        if old is None: return new
        return (1 - self.alpha) * old + self.alpha * new

    def _find_two_largest(self, contours):
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[:2] if len(contours) >= 2 else contours

    def _angle_conv(self, px, py, cx, cy):
        dx, dy = cx - px, cy - py
        ang = math.degrees(math.atan2(dy, dx))  # -180..180 wrt +x axis
        # Your earlier convention: fold to [-90, 90]
        if ang > 90: ang -= 180
        elif ang < -90: ang += 180

        # Enviar ángulo izquierdo por UDP si está disponible
        angL = ang  # ángulo izquierdo
        angL_rad = math.radians(angL)  # conversión a radianes
        payload = struct.pack('<d', angL_rad)  # float64
        sock.sendto(payload, UDP_ADDR_CMD)

        return ang

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        min_area = max(1, int(self.min_area_ratio * w * h))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        cnts = self._find_two_largest(cnts)

        centroids = []
        for c in cnts:
            x,y,wc,hc = cv2.boundingRect(c)
            cx, cy = x + wc//2, y + hc//2
            centroids.append(np.array([cx, cy], dtype=np.float32))
            cv2.rectangle(frame, (x,y), (x+wc, y+hc), (0,0,255), 2)
            cv2.circle(frame, (cx, cy), 8, (255,0,255), -1)

        if len(centroids) == 2:
            # order left/right by x
            centroids.sort(key=lambda p: p[0])
            cL, cR = centroids
            # smooth
            self.ema_centroids[0] = self._smooth(self.ema_centroids[0], cL)
            self.ema_centroids[1] = self._smooth(self.ema_centroids[1], cR)
            cL, cR = self.ema_centroids

            ref = (cL + cR) / 2.0
            self.ema_ref = self._smooth(self.ema_ref, ref) if self.ema_ref is not None else ref
            ref = self.ema_ref

            if self.fixed_ref is None:
                self.fixed_ref = ref  # guardar primera vez

            px, py = int(self.fixed_ref[0]), int(self.fixed_ref[1])
            cv2.line(frame, (px-350, py), (px+350, py), (0,255,0), 3)

            # angles vs reference
            angL = self._angle_conv(px, py, int(cL[0]), int(cL[1]))
            angR = self._angle_conv(px, py, int(cR[0]), int(cR[1]))

            # draw rays + labels
            for (cx, cy, ang, tag) in [(int(cL[0]), int(cL[1]), angL, "L"),
                                       (int(cR[0]), int(cR[1]), angR, "R")]:
                cv2.line(frame, (px, py), (cx, cy), (255,0,255), 2)
                mx, my = (px+cx)//2, (py+cy)//2
                cv2.putText(frame, f"{tag}:{ang:.1f}°", (mx, my-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            angles = [float(angL), float(angR)]
        else:
            angles = []

        return frame, angles

    def release(self):
        if self.cap: self.cap.release()

def main():
    # --- Option 1: Use camera (default index 0) ---
    # tracker = LiveAeroTracker(cam_index=0)

    # --- Option 2: Use video file instead of camera ---
    tracker = LiveAeroTracker(cam_index="prueba2.mp4")

    print("Controls: 'q' quit | 'h' show HSV sliders | 'p' print angles")
    show_tracks = False

    # optional realtime HSV tuning UI
    def nothing(_): pass
    def ensure_trackbars():
        cv2.namedWindow("HSV")
        cv2.createTrackbar("Hmin", "HSV", 20, 179, nothing)
        cv2.createTrackbar("Smin", "HSV", 100, 255, nothing)
        cv2.createTrackbar("Vmin", "HSV", 100, 255, nothing)
        cv2.createTrackbar("Hmax", "HSV", 35, 179, nothing)
        cv2.createTrackbar("Smax", "HSV", 255, 255, nothing)
        cv2.createTrackbar("Vmax", "HSV", 255, 255, nothing)

    while True:
        ok, frame = tracker.cap.read()
        if not ok: break

        if show_tracks:
            # read sliders
            lh = cv2.getTrackbarPos("Hmin","HSV"); ls = cv2.getTrackbarPos("Smin","HSV"); lv = cv2.getTrackbarPos("Vmin","HSV")
            uh = cv2.getTrackbarPos("Hmax","HSV"); us = cv2.getTrackbarPos("Smax","HSV"); uv = cv2.getTrackbarPos("Vmax","HSV")
            tracker.lower_yellow = np.array([lh, ls, lv])
            tracker.upper_yellow = np.array([uh, us, uv])

        annotated, angles = tracker.process_frame(frame)

        cv2.imshow("Aero Live", annotated)
        time.sleep(0.033)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            show_tracks = not show_tracks
            if show_tracks: ensure_trackbars()
            else: 
                try: cv2.destroyWindow("HSV")
                except: pass
        elif key == ord('p') and angles:
            print(f"Angles (L,R): {angles}")

    
    tracker.release()
    sock.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
