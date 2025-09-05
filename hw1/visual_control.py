import cv2, numpy as np, math, time
import socket, struct

# Configuración UDP
UDP_ADDR_CMD = ('127.0.0.1', 5005)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class LiveAeroTracker:
    def __init__(self, min_area_ratio=0.0005, alpha=0.3):
        # HSV ranges para amarillo
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        self.min_area_ratio = min_area_ratio
        self.alpha = alpha
        self.ema_ref = None
        self.ema_centroids = [None, None]
        self.fixed_ref = None

        self.last_sent_angle = None
        self.min_angle_change = 0.05  # rad (~3°)
        self.last_sent_time = time.time()
        self.send_interval = 0.05  # seconds

    def _smooth(self, old, new):
        if old is None: return new
        return (1 - self.alpha) * old + self.alpha * new

    def _find_two_largest(self, contours):
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[:2] if len(contours) >= 2 else contours

    def _angle_conv(self, px, py, cx, cy):
        dx, dy = cx - px, cy - py
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90: ang -= 180
        elif ang < -90: ang += 180

        ang_rad = math.radians(ang)
        now = time.time()

        if now - self.last_sent_time > self.send_interval:
            payload = struct.pack('<d', ang_rad)
            sock.sendto(payload, UDP_ADDR_CMD)
            self.last_sent_time = now

        if self.last_sent_angle is None or abs(ang_rad - self.last_sent_angle) > self.min_angle_change:
            payload = struct.pack('<d', ang_rad)
            sock.sendto(payload, UDP_ADDR_CMD)
            self.last_sent_angle = ang_rad

        return ang

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        min_area = max(1, int(self.min_area_ratio * w * h))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

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
            cv2.rectangle(frame, (x,y), (x+wc, y+hc), (0,0,255), 5)
            cv2.circle(frame, (cx, cy), 8, (255,0,255), -1)

        angles = []
        if len(centroids) == 2:
            centroids.sort(key=lambda p: p[0])
            cL, cR = centroids
            self.ema_centroids[0] = self._smooth(self.ema_centroids[0], cL)
            self.ema_centroids[1] = self._smooth(self.ema_centroids[1], cR)
            cL, cR = self.ema_centroids

            ref = (cL + cR) / 2.0
            self.ema_ref = self._smooth(self.ema_ref, ref) if self.ema_ref is not None else ref
            ref = self.ema_ref

            if self.fixed_ref is None:
                self.fixed_ref = ref

            px, py = int(self.fixed_ref[0]), int(self.fixed_ref[1])
            cv2.line(frame, (px-350, py), (px+350, py), (0,255,0), 5)

            angL = self._angle_conv(px, py, int(cL[0]), int(cL[1]))
            angR = self._angle_conv(px, py, int(cR[0]), int(cR[1]))

            for (cx, cy, ang, tag) in [(int(cL[0]), int(cL[1]), angL, "L"),
                                       (int(cR[0]), int(cR[1]), angR, "R")]:
                cv2.line(frame, (px, py), (cx, cy), (255,0,255), 5)
                mx, my = (px+cx)//2, (py+cy)//2
                cv2.putText(frame, f"{tag}:{ang:.1f}", (mx, my-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 4, cv2.LINE_AA)
            angles = [float(angL), float(angR)]

        return frame, angles


# --- GUI con botones dibujados ---
tracker = LiveAeroTracker()
current_img = None
angles = []

# Definir regiones de botones
button_A = (20, 20, 180, 70)   # x1,y1,x2,y2
button_B = (200, 20, 360, 70)

def mouse_callback(event, x, y, flags, param):
    global current_img, angles, tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_A[0] <= x <= button_A[2] and button_A[1] <= y <= button_A[3]:
            img = cv2.imread("posA.jpeg")
        elif button_B[0] <= x <= button_B[2] and button_B[1] <= y <= button_B[3]:
            img = cv2.imread("posB.jpeg")
        else:
            return

        if img is not None:
            # Resetear EMA y referencia al cambiar imagen
            tracker.ema_centroids = [None, None]
            tracker.ema_ref = None
            tracker.fixed_ref = None

            current_img, angles = tracker.process_frame(img)

cv2.namedWindow("Aero GUI")
cv2.setMouseCallback("Aero GUI", mouse_callback)

while True:
    # Fondo base
    display = np.ones((600, 800, 3), dtype=np.uint8) * 30

    # Dibujar botones
    cv2.rectangle(display, (button_A[0], button_A[1]), (button_A[2], button_A[3]), (0,255,255), -1)
    cv2.putText(display, "Imagen A", (button_A[0]+20, button_A[1]+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.rectangle(display, (button_B[0], button_B[1]), (button_B[2], button_B[3]), (0,255,255), -1)
    cv2.putText(display, "Imagen B", (button_B[0]+20, button_B[1]+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    # Si hay imagen procesada, mostrarla abajo
    if current_img is not None:
        h, w = current_img.shape[:2]
        scale = 600 / max(h, w)
        resized = cv2.resize(current_img, (int(w*scale), int(h*scale)))
        display[100:100+resized.shape[0], 50:50+resized.shape[1]] = resized

        cv2.putText(display, f"Angles: {angles}", (50, 580),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Aero GUI", display)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
sock.close()
