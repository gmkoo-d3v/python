#!/bin/bash

file="robot.py"

if [ -f "$file" ]; then          # -f : 파일이 존재하면
    echo "$file 있음"
else
    echo "$file 없음"
fi                               # if 의 끝 (fi)

# 숫자 비교
count=2
if [ $count -gt 5 ]; then        # -gt : greater than (>)
    echo "5보다 큼"
elif [[ $count -lt 10 && $count -gt 5 ]]; then   # -lt : less than (<)
    echo "less than 10 but more than 5"

# 5보다 작음 
else
    echo "less than 5"
fi
