#!/bin/bash

#1013/test.sh 만들기
#디렉토리 갯수 세기
dir_count=$(find . -type d | wc -l)
today=$(date +%Y%m%d)
test_dir="test_$today"


echo "현재 폴더내의 하위 폴더 갯수: $dir_count개"

if [ ! -d "test_dir"];then
   echo "$test_dir 생성"
   mkdir "$test_dir"
else
   echo "$test_dir already exist"
fi


# txt 갯수 세기
txt_count=0
for f in *.txt; do
   [ -f "$f" ] || continue
   cp "$f" "$test_dir/"
   txt_count=$((txt_count+1))

done
echo "현재 폴더내의 하위 텍스트 갯수: $txt_count개"


#반복문으로 .txt파일 test_오늘날짜 디렉토리 만들어 옮기기

#str1='date'
#str2='test_'
#temp="$str2$str1"
#mkdir $temp
#mv *.txt ./$temp


echo "완료! 디렉토리수 : $dir_count 텍스트 수: $txt_count"

#ls 코드 수정 필요
ls "$test_dir"
