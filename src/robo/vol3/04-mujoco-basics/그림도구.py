"""4장: 렌더링한 프레임 여러 장을 한 그림에 나란히 놓는 도구."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "D2Coding"
plt.rcParams["axes.unicode_minus"] = False


def 프레임격자(프레임들: list[np.ndarray], 제목들: list[str], 파일: str, 열: int = 3) -> None:
    """이미지 여러 장을 행×열 격자로 배치해 파일로 저장한다."""
    행 = (len(프레임들) + 열 - 1) // 열
    높이, 너비 = 프레임들[0].shape[:2]
    그림, 칸들 = plt.subplots(행, 열, figsize=(2.6 * 열, 2.6 * 행 * 높이 / 너비 + 0.3 * 행))
    for 칸, 프레임, 제목 in zip(np.ravel(칸들), 프레임들, 제목들):
        칸.imshow(프레임)
        칸.set_title(제목, fontsize=10)
    for 칸 in np.ravel(칸들):
        칸.axis("off")
    그림.tight_layout()
    그림.savefig(파일, dpi=200)
    plt.close(그림)
