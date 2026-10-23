import cv2
import mediapipe as mp
import numpy as np

print(f"Python: {__import__('sys').version.split()[0]}")
print(f"OpenCV: {cv2.__version__}")
print(f"MediaPipe: {mp.__version__}")
print(f"NumPy: {np.__version__}")

# 간단한 MediaPipe Hands 테스트 (이미지 없이 파이프라인 생성)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
print("MediaPipe Hands 생성 성공")

# 더미 이미지로 테스트
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
results = hands.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
print(f"더미 이미지 처리 성공: {results.multi_hand_landmarks}")

hands.close()
print("모든 테스트 통과!")
