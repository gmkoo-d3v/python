#!/bin/bash
# 로그 폴더를 날짜별로 백업하는 스크립트

source_dir="logs"
today=$(date +%Y%m%d)          # $(...) : 명령 실행 결과를 변수에 담는 '명령 치환'
backup_dir="backup_$today"

# 원본 폴더가 있는지 확인 (조건문)
if [ ! -d "$source_dir" ]; then    # ! -d : 폴더가 없으면
    echo "오류: $source_dir 폴더가 없습니다"
    exit 1                     # 스크립트 중단
fi

# 백업 폴더 생성 (1장 mkdir)
mkdir -p "$backup_dir"

# .log 파일들을 복사 (반복문 + 1장 cp)
count=0
for f in $source_dir/*.log; do
    [ -f "$f" ] || continue    # .log 파일이 하나도 없을 때(빈 glob) 건너뜀
    cp "$f" "$backup_dir/"
    echo "백업: $f"
    count=$((count + 1))       # 숫자 계산
done

echo "완료! $count개 파일을 $backup_dir 에 백업했습니다"
