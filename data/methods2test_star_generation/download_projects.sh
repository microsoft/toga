#!/bin/bash
mkdir -p methods2test-projects/
cd methods2test-projects/
input="../repo_list.txt"
while IFS= read -r project
do
  if [ ! -z "$project" ]
  then
    echo "$project"
    GIT_TERMINAL_PROMPT=0 git clone --depth=1 --single-branch $project 
  fi
done < "$input"
