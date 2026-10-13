#!/bin/bash

# 목록 순회 (do ... done)
for name in 철수 영희 민수; do
    echo "안녕, $name"
done

# 파일들 순회 — 모든 .txt 파일 처리
for f in *.txt; do
    echo "처리 중: $f"
    echo "hel" >> "$f"
done

# 숫자 범위 (1부터 5까지)
for i in {1..5}; do
    echo "반복 $i"
done
