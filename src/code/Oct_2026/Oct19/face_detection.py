import cv2
import mediapipe as mp

# MediaPipe 얼굴 감지 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('카메라에서 프레임을 가져올 수 없습니다.')
            break

        # 이미지 전처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # 결과 표시
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('MediaPipe 얼굴 인식', image)

        if cv2.waitKey(5) & 0xFF == 27:  # esc 로 종료
            break

cap.release()
cv2.destroyAllWindows()
