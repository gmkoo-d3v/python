import time

주기 = 0.02                       # 50Hz: 1초에 50번
박자_수 = 10

시작_시각 = time.perf_counter()
다음_박자 = 시작_시각
for 박자 in range(박자_수):
    # 여기에 감지 → 판단 → 행동이 들어간다(이 예에서는 할 일이 없다)
    다음_박자 += 주기                                  # 다음 박자의 시각을 미리 정해 두고
    남은_시간 = 다음_박자 - time.perf_counter()
    if 남은_시간 > 0:
        time.sleep(남은_시간)                           # 그때까지만 쉰다
    else:
        print(f"{박자}번째 박자를 놓쳤다")               # 할 일이 주기보다 오래 걸렸다
print(f"{박자_수}박자에 약 {time.perf_counter() - 시작_시각:.1f}초")
