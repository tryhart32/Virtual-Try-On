import cv2, numpy as np, mediapipe as mp

def overlay(bg, fg, x, y, size=None):
    if size: fg = cv2.resize(fg, size)
    b, g, r, a = cv2.split(fg)
    mask = a / 255.0
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]: return bg
    roi = bg[y:y+h, x:x+w]
    blended = roi * (1-mask[...,None]) + cv2.merge((b,g,r)) * mask[...,None]
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    return bg

shirts = [cv2.imread(f"{i}.png", cv2.IMREAD_UNCHANGED) for i in range(1, 4)]
btn_L = cv2.imread("button_left.png", cv2.IMREAD_UNCHANGED)
btn_R = cv2.imread("button_right.png", cv2.IMREAD_UNCHANGED)

mp_pose, mp_hands = mp.solutions.pose, mp.solutions.hands
pose, hands = mp_pose.Pose(), mp_hands.Hands()
cap, idx = cv2.VideoCapture(0), 0
was_L = was_R = False

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose untuk kaos
    res_p = pose.process(rgb)
    if res_p.pose_landmarks:
        lm = res_p.pose_landmarks.landmark
        l_s, r_s = lm[11], lm[12]
        l_h, r_h = lm[23], lm[24]
        x1, x2 = int(l_s.x*w), int(r_s.x*w)
        y1, y2 = int(min(l_s.y, r_s.y)*h), int(max(l_h.y, r_h.y)*h)
        sw, sh = abs(x2-x1)+50, abs(y2-y1)+30
        frame = overlay(frame, shirts[idx], (x1+x2)//2-sw//2, y1-30, (sw, sh))

    # Tombol kiri kanan
    pos_L, pos_R = (20, h//2-32), (w-84, h//2-32)
    frame = overlay(frame, btn_L, *pos_L, (64, 64))
    frame = overlay(frame, btn_R, *pos_R, (64, 64))

    # Tangan & sentuhan tombol
    res_h = hands.process(rgb)
    L, R = False, False
    if res_h.multi_hand_landmarks:
        for hand in res_h.multi_hand_landmarks:
            fx, fy = int(hand.landmark[8].x*w), int(hand.landmark[8].y*h)
            cv2.circle(frame, (fx, fy), 10, (0,255,255), -1)
            L = pos_L[0] < fx < pos_L[0]+64 and pos_L[1] < fy < pos_L[1]+64
            R = pos_R[0] < fx < pos_R[0]+64 and pos_R[1] < fy < pos_R[1]+64
            break

    if L and not was_L: idx = (idx-1)%len(shirts)
    if R and not was_R: idx = (idx+1)%len(shirts)
    was_L, was_R = L, R

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()