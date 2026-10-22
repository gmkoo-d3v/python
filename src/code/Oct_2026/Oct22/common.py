"""
부록 A 공용 도구 — 그림 스타일, 저장 위치, 표 출력.

슬라이드의 숫자와 그림은 전부 이 폴더의 스크립트가 만든 실측값이다.
스크립트를 다시 돌리면 같은 시드로 같은 숫자가 나온다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 PNG만 만든다 (Jetson·서버에서도 돈다)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

FIG_DIR = Path(__file__).resolve().parent.parent / "figs"
SEED = 0

# 교재 비전 과목 색 — 앱의 강조색과 상자 색을 그대로 쓴다
TEAL = "#2A9D8F"
TEAL_DARK = "#218075"
ORANGE = "#B3541E"
GOLD = "#D9A02B"
INK = "#1C2733"
INK_SOFT = "#47576B"
LINE = "#E2DDD0"
# 결정 경계 그림의 배경 — 주황·청록을 아주 옅게 깐 색
ORANGE_PALE = "#F6E3D8"
TEAL_PALE = "#D7EEEA"

# 맥·우분투·윈도우에서 흔한 한글 글꼴을 차례로 찾는다
KOREAN_FONTS = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"]
FIG_DPI = 200
FIG_SIZE = (7.2, 4.0)
WIDE_VALUE = 10  # 표에서 절댓값이 이 이상이면 소수 한 자리만 쓴다


def setup_style() -> None:
    """한글 글꼴과 교재 색을 matplotlib 기본값으로 둔다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((name for name in KOREAN_FONTS if name in installed), None)
    if family:
        plt.rcParams["font.family"] = family
    plt.rcParams.update({
        "axes.unicode_minus": False,  # 한글 글꼴에서 마이너스가 네모로 깨지는 것을 막는다
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": LINE,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "figure.figsize": FIG_SIZE,
    })


def log_axis(ax: plt.Axes) -> None:
    """y축을 로그 눈금으로 두되 눈금 글자는 0.01·10 처럼 그냥 숫자로 쓴다.
    기본 표기(10⁻²)는 수식 글꼴을 거치는데 한글 글꼴에는 그 마이너스 기호가 없다."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())  # 보조 눈금(6×10⁻¹)도 수식 표기라 글자를 지운다


def log_x(ax: plt.Axes) -> None:
    """x축 로그 눈금 — log_axis 와 같은 이유로 눈금 글자를 그냥 숫자로 쓴다."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig: plt.Figure, name: str) -> Path:
    """figs/ 아래에 PNG로 저장하고 경로를 돌려준다."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  그림 저장: figs/{path.name}")
    return path


def table(headers: list[str], rows: list[list[object]]) -> None:
    """슬라이드에 옮겨 적을 표를 고정폭으로 찍는다."""
    cells = [headers] + [[_fmt(v) for v in row] for row in rows]
    widths = [max(len(str(r[i])) for r in cells) for i in range(len(headers))]
    for i, row in enumerate(cells):
        print("  " + "  ".join(str(v).rjust(w) for v, w in zip(row, widths)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < WIDE_VALUE else f"{value:.1f}"
    return str(value)
