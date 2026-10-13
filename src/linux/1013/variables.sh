#!/bin/bash

name="철수"              # 저장 (= 양옆 공백 없이!)
count=5

echo "이름: $name"       # 사용할 땐 $ 붙임
echo "개수: $count"
echo "폴더: ${name}_backup"   # ${} 로 감싸면 명확
