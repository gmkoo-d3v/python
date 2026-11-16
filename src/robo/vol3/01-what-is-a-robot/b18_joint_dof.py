import mujoco

# 관절 종류 네 가지를 하나씩 단 몸체. nq는 자세를 적는 숫자 수, nv는 움직일 수 있는 방향 수(자유도)
관절_종류 = {"hinge(경첩)": '<joint type="hinge"/>', "slide(미닫이)": '<joint type="slide"/>',
           "ball(공)": '<joint type="ball"/>', "free(자유)": "<freejoint/>"}
for 이름, 관절 in 관절_종류.items():
    xml = f"""<mujoco><worldbody><body>{관절}
              <geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
    모델 = mujoco.MjModel.from_xml_string(xml)
    print(f"{이름}: 숫자 nq={모델.nq}, 자유도 nv={모델.nv}")
