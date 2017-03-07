#!/bin/sh
BASEURL="_posts/"
NOW=$(date +"%Y-%m-%d")
if [ $# -gt 1 ]; then
    NOW=$2
fi
HYPHEN="-"
EXT=".markdown"
FILE=$BASEURL$NOW$HYPHEN${1// /-}$EXT
echo $FILE" created!"
touch $FILE
echo "---" >> $FILE
echo "layout: post"  >> $FILE
echo "title: \""$1"\""  >> $FILE
echo "date: "$NOW  >> $FILE
echo "mathjax: true" >> $FILE
echo "categories: "  >> $FILE
echo "tags: " >> $FILE
echo "---" >> $FILE
echo "* content" >> $FILE
echo "{:toc}" >> $FILE
