import cv2
import mediapipe as mp
import pygame
import math
import time

pygame.mixer.init()
pygame.mixer.set_num_channels(32)  # allow many overlapping sounds

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load sounds for left and right hands
sounds_left = {
    "Db4": pygame.mixer.Sound("Piano.ff.Db4.wav"),
    "Eb4": pygame.mixer.Sound("Piano.ff.Eb4.wav"),
    "F4": pygame.mixer.Sound("Piano.ff.F4.wav"),
    "Gb4": pygame.mixer.Sound("Piano.ff.Gb4.wav")
}

sounds_right = {
    "Ab4": pygame.mixer.Sound("Piano.ff.Ab4.wav"),
    "Bb4": pygame.mixer.Sound("Piano.ff.Bb4.wav"),
    "B4": pygame.mixer.Sound("Piano.ff.B4.wav"),
    "C4": pygame.mixer.Sound("Piano.ff.C4.wav")
}

# Cooldown dictionary
last_play_time = {note: 0 for note in list(sounds_left.keys()) + list(sounds_right.keys())}
cooldown = 0.3

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 800)
while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = img.shape
    cv2.rectangle(img, (340, 30), (940, 110), (255, 0, 0), -1)
    cv2.putText(img, "Gesture Piano", (470, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.rectangle(img, (30, 30), (300, 110), (0, 255, 0), -1)
    cv2.putText(img, "Left Hand", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "i=Db4,m=F4,r=Eb4,p=Gb4", (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.rectangle(img, (980, 30), (1250, 110), (0, 255, 0), -1)
    cv2.putText(img, "Right Hand", (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "i=Ab4,m=Bb4,r=B4,p=C4", (1000, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_label.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb and finger tip landmarks
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            ring = hand_landmarks.landmark[16]
            pink = hand_landmarks.landmark[20]

            # find axis of finger tips
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)
            middle_x, middle_y = int(middle.x * w), int(middle.y * h)
            ring_x, ring_y = int(ring.x * w), int(ring.y * h)
            pink_x, pink_y = int(pink.x * w), int(pink.y * h)

            # mid point between thumb and fingers
            index_midx, index_midy = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
            middle_midx, middle_midy = (thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2
            ring_midx, ring_midy = (thumb_x + ring_x) // 2, (thumb_y + ring_y) // 2
            pink_midx, pink_midy = (thumb_x + pink_x) // 2, (thumb_y + pink_y) // 2

            # draw circles
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), -1)
            cv2.circle(img, (middle_x, middle_y), 10, (255, 0, 0), -1)
            cv2.circle(img, (ring_x, ring_y), 10, (255, 0, 0), -1)
            cv2.circle(img, (pink_x, pink_y), 10, (255, 0, 0), -1)

            # Draw line all fingers to thumb finger
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            cv2.line(img, (thumb_x, thumb_y), (middle_x, middle_y), (0, 255, 0), 2)
            cv2.line(img, (thumb_x, thumb_y), (ring_x, ring_y), (0, 255, 0), 2)
            cv2.line(img, (thumb_x, thumb_y), (pink_x, pink_y), (0, 255, 0), 2)

            # Draw mid point from the thumb finger
            cv2.circle(img, (index_midx, index_midy), 10, (255, 0, 255), -1)
            cv2.circle(img, (middle_midx, middle_midy), 10, (255, 0, 255), -1)
            cv2.circle(img, (ring_midx, ring_midy), 10, (255, 0, 255), -1)
            cv2.circle(img, (pink_midx, pink_midy), 10, (255, 0, 255), -1)

            # Distances
            l_index = math.hypot(index_x - thumb_x, index_y - thumb_y)
            l_middle = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
            l_ring = math.hypot(ring_x - thumb_x, ring_y - thumb_y)
            l_pink = math.hypot(pink_x - thumb_x, pink_y - thumb_y)

            # Assign note set
            sounds = sounds_left if label == "Left" else sounds_right

            # Check each finger
            if l_index < 30 and (time.time() - last_play_time[list(sounds.keys())[0]]) > cooldown:
                cv2.circle(img, (index_midx, index_midy), 10, (0, 255, 0), -1)
                list(sounds.values())[0].play()
                last_play_time[list(sounds.keys())[0]] = time.time()

            if l_middle < 30 and (time.time() - last_play_time[list(sounds.keys())[1]]) > cooldown:
                cv2.circle(img, (middle_midx, middle_midy), 10, (0, 255, 0), -1)
                list(sounds.values())[1].play()
                last_play_time[list(sounds.keys())[1]] = time.time()

            if l_ring < 30 and (time.time() - last_play_time[list(sounds.keys())[2]]) > cooldown:
                cv2.circle(img, (ring_midx, ring_midy), 10, (0, 255, 0), -1)
                list(sounds.values())[2].play()
                last_play_time[list(sounds.keys())[2]] = time.time()

            if l_pink < 30 and (time.time() - last_play_time[list(sounds.keys())[3]]) > cooldown:
                cv2.circle(img, (pink_midx, pink_midy), 10, (0, 255, 0), -1)
                list(sounds.values())[3].play()
                last_play_time[list(sounds.keys())[3]] = time.time()

    cv2.imshow("Virtual Band", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()