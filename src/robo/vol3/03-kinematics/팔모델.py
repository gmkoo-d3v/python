"""팔도구.py의 예제_팔을 MuJoCo 모델 글(MJCF)로 옮긴다. MJCF 문법은 4장에서 자세히 배운다."""
from 팔도구 import 손끝_위치, 예제_팔


def 글(벡터) -> str:
    return " ".join(str(float(값)) for 값 in 벡터)


def MJCF(팔=예제_팔, 손끝=손끝_위치) -> str:
    """관절마다 body 하나: 부모 기준 위치(pos), 회전축(joint), 다음 관절까지 막대(geom)."""
    몸체들 = []
    for i, (_, 위치, 축) in enumerate(팔):
        다음 = 팔[i + 1][1] if i + 1 < len(팔) else 손끝
        몸체들.append(f'<body name="link{i + 1}" pos="{글(위치)}">'
                    f'<joint name="joint{i + 1}" type="hinge" axis="{글(축)}"/>'
                    f'<geom type="capsule" fromto="0 0 0 {글(다음)}" size="0.02" rgba="0.8 0.5 0.2 1"/>')
    끝 = f'<site name="tip" pos="{글(손끝)}" size="0.015" rgba="0.1 0.3 0.9 1"/>'
    return f"""
<mujoco>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <geom type="cylinder" fromto="0 0 0 {글(팔[0][1])}" size="0.05" rgba="0.3 0.3 0.3 1"/>
    {"".join(몸체들)}{끝}{"</body>" * len(팔)}
  </worldbody>
</mujoco>"""
