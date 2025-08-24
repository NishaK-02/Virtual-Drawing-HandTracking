import cv2
import numpy as np
import mediapipe as mp
import random
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0
color = (255, 0, 0)
brush_size = 4
random_color = False
draw_mode = True
eraser = False

undo_stack = deque(maxlen=20)
redo_stack = deque()
save_cooldown = False  # To prevent multiple saves
finger_tips = [4, 8, 12, 16, 20]

def fingers_up(hand):
    lm = hand.landmark
    fingers = []
    fingers.append(lm[4].x < lm[3].x)  # Thumb
    for tip in finger_tips[1:]:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers

def is_open_palm(fingers):
    return fingers.count(True) >= 4

def is_thumbs_up(fingers):
    return fingers == [True, False, False, False, False]

def save_state():
    if canvas is not None:
        undo_stack.append(canvas.copy())
        redo_stack.clear()

print("Controls:")
print("  r/g/b/k/w: Change color")
print("  + / -: Brush size")
print("  m: Random color")
print("  d: Toggle draw mode")
print("  z/y: Undo / Redo")
print("  c: Clear")
print("  Esc: Exit")
print("  👋 Open Palm = Eraser | 👍 Thumbs Up = Save")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS, 
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))

            h, w, _ = frame.shape
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            fingers = fingers_up(handLms)

            if is_open_palm(fingers):
                eraser = True
                gesture_text = "Eraser (Open Palm)"
            elif is_thumbs_up(fingers):
                if not save_cooldown:
                    cv2.imwrite("drawing.png", canvas)
                    print("👍 Drawing saved as drawing.png")
                    gesture_text = "Saved (Thumbs Up)"
                    save_cooldown = True
            else:
                eraser = False
                save_cooldown = False  # Reset cooldown when not showing thumbs up

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            if random_color:
                draw_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            elif eraser:
                draw_color = (0, 0, 0)
            else:
                draw_color = color

            if draw_mode and not is_thumbs_up(fingers):
                save_state()
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0
        save_cooldown = False

    out = cv2.add(frame, canvas)

    preview_color = (0, 0, 0) if eraser else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if random_color else color
    cv2.rectangle(out, (10, 50), (60, 100), preview_color, -1)
    cv2.putText(out, f"Brush: {brush_size}px", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, f"{'Eraser' if eraser else 'Color'}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, f"Drawing: {'ON' if draw_mode else 'OFF'}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if gesture_text:
        cv2.putText(out, gesture_text, (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.imshow("Virtual Drawing Canvas", out)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        color = (0, 0, 255)
        random_color = False
        eraser = False
    elif key == ord('g'):
        color = (0, 255, 0)
        random_color = False
        eraser = False
    elif key == ord('b'):
        color = (255, 0, 0)
        random_color = False
        eraser = False
    elif key == ord('k'):
        color = (0, 0, 0)
        random_color = False
        eraser = False
    elif key == ord('w'):
        color = (255, 255, 255)
        random_color = False
        eraser = False
    elif key == ord('+') or key == ord('='):
        brush_size += 1
    elif key == ord('-') or key == ord('_'):
        brush_size = max(1, brush_size - 1)
    elif key == ord('m'):
        random_color = not random_color
        eraser = False
    elif key == ord('d'):
        draw_mode = not draw_mode
    elif key == ord('z'):
        if undo_stack:
            redo_stack.append(canvas.copy())
            canvas = undo_stack.pop()
    elif key == ord('y'):
        if redo_stack:
            undo_stack.append(canvas.copy())
            canvas = redo_stack.pop()
    elif key == ord('c'):
        save_state()
        canvas[:] = 0
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
