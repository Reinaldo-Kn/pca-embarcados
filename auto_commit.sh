#!/bin/bash 

current_version=$(cat VERSION)

#versão em parte ( major, minor, patch)
IFS='.' read -r major minor patch <<< "$current_version"

patch=$((patch+1))

new_version="$major.$minor.$patch"

#atualiza o arquivo VERSION
echo $new_version > VERSION

# commit automatico 
git add .
git commit -m "v$new_version"
git push origin master

#print nova versao
echo "Nova versão: $new_version"