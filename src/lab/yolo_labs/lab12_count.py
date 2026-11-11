"""11차시: 검출만으로는 사람을 셀 수 없다.

정답 ID가 있는 합성 영상(사람 12명, 16초)을 만들고, 2일차 best.pt로 장면마다 사람을 찾는다.
장면마다 찾은 수를 더하거나, 가장 많이 찾은 장면을 고르는 식으로는 '몇 명이 지나갔나'를 답할 수 없다.
lab5를 먼저 돌려 runs/person/weights/best.pt가 있어야 합니다.
"""
import numpy as np

from common_track import FPS, FRAMES, PEOPLE, VIDEO, synth, synth_detections

CONF = 0.5


def main() -> None:
    frames, truth = synth()
    dets = synth_detections()
    per_frame = [int((d[:, 4] >= CONF).sum()) for d in dets]
    truth_per_frame = [len(t) for t in truth]
    print(f"합성 영상: {len(frames)}장면, {FPS}fps({FRAMES / FPS:.0f}초), 실제로 지나간 사람 {PEOPLE}명 → {VIDEO.name}")
    print(f"장면마다 보이는 사람(정답): 평균 {np.mean(truth_per_frame):.1f}명, 가장 많을 때 {max(truth_per_frame)}명")
    print(f"장면마다 찾은 사람(conf {CONF}): 평균 {np.mean(per_frame):.1f}명, 가장 많을 때 {max(per_frame)}명")
    print()
    print("'몇 명이 지나갔나'를 검출만으로 답해 보기")
    print(f"  장면마다 찾은 수를 모두 더하기 : {sum(per_frame)}명")
    print(f"  가장 많이 찾은 장면의 수       : {max(per_frame)}명")
    print(f"  정답                          : {PEOPLE}명")
    print()
    print("같은 사람이 장면마다 다시 세지고, 한 장면에는 모두가 함께 나오지 않는다.")
    print("'이 상자와 저 상자가 같은 사람인가'를 이어 주는 일, 그것이 추적이다.")


if __name__ == "__main__":
    main()
