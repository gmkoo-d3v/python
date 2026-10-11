# nums = [1,2,3,3,2,1,4,5,6]
# nums.sort(reverse=True)
# print(nums)
# print(sum(nums))
# print(max(nums), min(nums))
# print(nums.count(1))
# print(nums.index(4))

# print("find")
# cnt = 0

# for i in nums:
#     print(i)
#     if i == 1:
#         cnt +=1
# print("1의 갯수", cnt)

# scores = {"철수": 90, "영희": 85}
# scores["영희"] = 100
# print(scores)

a = [1,2]
b = a[:]
b.append(3)

n = 0
if n % 2 == 0:
    print(f"{n}은 짝수")
elif n == 0:
    print(f"{n}은 0")
else:
    print(f"{n}은 홀수")

score = print("점수입력:"),int(input())

if score >= 90:
    grade = "a"
elif score >= 80:
    grade = "b"
elif score >= 70:
    grade = "c"
else:
    grade = "f"
print(f"점수 {score}->학점{grade}")    