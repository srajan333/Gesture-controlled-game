#!/usr/bin/env python3
"""
Hand Gesture Car Control (Final Improved Version)
-------------------------------------------------
Technologies: OpenCV, MediaPipe, PyAutoGUI, NumPy

Gestures → Actions:
 - Left Hand  (Open/Point)   → Accelerate (↑)
 - Right Hand (Open/Point)   → Brake (↓)
 - Move Hand Left/Right      → Steering (← →)
 - Both Hands Close Together → Nitro Boost (Space)
 - Fist                     → Idle (no action)

Extras:
 - Live status HUD
 - On-screen legend (gesture guide box)
 - Smooth steering (smoothing buffer)
 - Clean key handling (no stuck keys)
"""

import cv2, mediapipe as mp, numpy as np, pyautogui, math
from collections import deque

# =================== CONFIG ===================
CONFIG = {
    "WEBCAM_INDEX": 0,
    "FRAME_WIDTH": 1280,
    "FRAME_HEIGHT": 720,
    "MIN_DETECTION_CONF": 0.6,
    "MIN_TRACKING_CONF": 0.6,
    "STEER_THRESHOLD": 60,      # px offset from center to trigger steering
    "SMOOTHING": 5,             # avg buffer size
    "NITRO_DISTANCE": 0.07,     # normalized hand distance for nitro
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
pyautogui.FAILSAFE = False

# =================== HELPERS ===================
def fingers_up(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    tips, mids = [4,8,12,16,20], [3,6,10,14,18]
    up=[False]*5
    for i in range(1,5):
        up[i]=pts[tips[i]][1]<pts[mids[i]][1]
    up[0]=abs(pts[4][0]-pts[3][0])>0.03
    return up

def classify_gesture(lm):
    f=fingers_up(lm); c=sum(f)
    if c>=2: return "open_palm"
    if c==0: return "fist"
    if f[1] and not any(f[2:]): return "point"
    return "unknown"

def normalized_distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

# =================== KEY CONTROL ===================
class KeyController:
    mapping={"accel":"up","brake":"down","left":"left","right":"right","nitro":"space"}
    def __init__(self): self.pressed=set()
    def press(self,a):
        if a not in self.pressed: pyautogui.keyDown(self.mapping[a]); self.pressed.add(a)
    def release(self,a):
        if a in self.pressed: pyautogui.keyUp(self.mapping[a]); self.pressed.remove(a)
    def tap(self,a): pyautogui.press(self.mapping[a])
    def release_all(self): [self.release(a) for a in list(self.pressed)]

# =================== DRAW GUIDE ===================
def draw_guide(frame):
    """Draw legend box with gesture→action mapping."""
    h,w=frame.shape[:2]
    box_w,box_h=360,180
    x0,y0=w-box_w-20,20
    cv2.rectangle(frame,(x0,y0),(x0+box_w,y0+box_h),(40,40,40),-1)
    cv2.putText(frame,"Gesture Guide",(x0+10,y0+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    guides=[
        ("Left Open/Point","Accelerate (↑)"),
        ("Right Open/Point","Brake (↓)"),
        ("Move Hand Left/Right","Steer (← →)"),
        ("Both Hands Close","Nitro (Space)"),
        ("Fist","Idle"),
    ]
    for i,(g,a) in enumerate(guides):
        cv2.putText(frame,f"{g} -> {a}",(x0+10,y0+55+i*25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

# =================== MAIN ===================
def main():
    cap=cv2.VideoCapture(CONFIG["WEBCAM_INDEX"])
    cap.set(3,CONFIG["FRAME_WIDTH"]); cap.set(4,CONFIG["FRAME_HEIGHT"])
    hands=mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIG["MIN_DETECTION_CONF"],
        min_tracking_confidence=CONFIG["MIN_TRACKING_CONF"]
    )
    keyctl=KeyController()
    left_buf,right_buf=deque(maxlen=CONFIG["SMOOTHING"]),deque(maxlen=CONFIG["SMOOTHING"])

    print("🚗 Car Gesture Control Started (press Q to quit)")
    try:
        while True:
            ret,frame=cap.read(); 
            if not ret: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=hands.process(rgb)

            accel=brake=left=right=nitro=False
            h,w=frame.shape[:2]; metas={}

            if res.multi_hand_landmarks:
                for lm,handed in zip(res.multi_hand_landmarks,res.multi_handedness):
                    label=handed.classification[0].label  # "Left"/"Right"
                    gesture=classify_gesture(lm.landmark)
                    cx,cy=np.mean([[p.x*w,p.y*h] for p in lm.landmark],axis=0)
                    metas[label]={"g":gesture,"c":(cx,cy)}
                    mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)

                # --- Left hand ---
                if "Left" in metas:
                    if metas["Left"]["g"] in ["open_palm","point"]: accel=True
                    left_buf.append(metas["Left"]["c"][0])
                    avg=np.mean(left_buf)
                    if avg<w/2-CONFIG["STEER_THRESHOLD"]: left=True
                    if avg>w/2+CONFIG["STEER_THRESHOLD"]: right=True
                # --- Right hand ---
                if "Right" in metas:
                    if metas["Right"]["g"] in ["open_palm","point"]: brake=True
                    right_buf.append(metas["Right"]["c"][0])
                    avg=np.mean(right_buf)
                    if avg<w/2-CONFIG["STEER_THRESHOLD"]: left=True
                    if avg>w/2+CONFIG["STEER_THRESHOLD"]: right=True
                # --- Nitro ---
                if "Left" in metas and "Right" in metas:
                    l,r=metas["Left"]["c"],metas["Right"]["c"]
                    if normalized_distance((l[0]/w,l[1]/h),(r[0]/w,r[1]/h))<CONFIG["NITRO_DISTANCE"]:
                        nitro=True

            # Apply keys
            keyctl.press("accel") if accel else keyctl.release("accel")
            keyctl.press("brake") if brake else keyctl.release("brake")
            if left: keyctl.press("left"); keyctl.release("right")
            elif right: keyctl.press("right"); keyctl.release("left")
            else: keyctl.release("left"); keyctl.release("right")
            if nitro: keyctl.tap("nitro")

            # HUD
            status=f"A:{accel} B:{brake} L:{left} R:{right} N:{nitro}"
            cv2.putText(frame,status,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            draw_guide(frame)

            cv2.imshow("Car Gesture Control",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    finally:
        print("Exiting, releasing keys...")
        keyctl.release_all(); cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()
