#!/usr/bin/env bash
# luna-filemap: a real file map of a directory, not a guess.
#   install -m755 scripts/file_map.sh ~/.local/bin/luna-filemap
#   luna-filemap [DIR] [-o FILE]     (default DIR=. ; -o writes Markdown to FILE)
# Lists every file under DIR (junk dirs skipped) with size, line count and a one-line
# purpose taken from the file's own header comment, grouped by directory.
set -euo pipefail

dir="."; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) dir="$1"; shift ;;
  esac
done
[ -d "$dir" ] || { echo "luna-filemap: not a directory: $dir" >&2; exit 1; }
cd "$dir"

purpose() {  # first useful comment/docstring line of a file, trimmed
  local f="$1"
  head -n 25 -- "$f" 2>/dev/null | LC_ALL=C grep -a -v '^\s*$' \
    | LC_ALL=C grep -a -v -E '^\s*(#!|<\?xml|<!DOCTYPE|---\s*$|\{\s*$|import |from |use |package |#include|set -|pragma)' \
    | LC_ALL=C grep -a -m1 -E '^\s*(//[/!]?|#|--|;;|/\*+|\*|"""|'"'''"'|<!--|%)' \
    | sed -E 's@^\s*(//[/!]?|#+|--|;;|/\*+|\*+|"""|'"'''"'|<!--|%)\s*@@; s@\s*(\*/|-->)\s*$@@' \
    | cut -c1-90 || true
}

gen() {
  echo "# File map: $(basename "$(pwd)")"
  echo
  total=$(find . \( -name .git -o -name node_modules -o -name target -o -name __pycache__ -o -name .venv -o -name dist \) -prune -o -type f -print | wc -l)
  echo "$total files. Generated $(date '+%Y-%m-%d %H:%M') by luna-filemap."
  echo
  find . \( -name .git -o -name node_modules -o -name target -o -name __pycache__ -o -name .venv -o -name dist \) -prune -o -type f -print \
    | sed 's@^\./@@' | LC_ALL=C sort | awk -F/ '{d=($0 ~ /\//)?$0:"."; sub(/\/[^\/]*$/,"",d); if(!($0 ~ /\//))d="."; print d "\t" $0}' \
    | while IFS=$'\t' read -r d f; do
        if [ "${last:-__none__}" != "$d" ]; then
          echo; echo "## $d/"; echo
          echo "| file | size | lines | purpose |"; echo "|---|---|---|---|"
          last="$d"
        fi
        size=$(du -h --apparent-size -- "$f" 2>/dev/null | cut -f1)
        if [ "$(head -c 4096 -- "$f" | tr -d '\000' | wc -c)" = "$(head -c 4096 -- "$f" | wc -c)" ]; then
          lines=$(wc -l < "$f"); p=$(purpose "$f" | sed 's/|/\\|/g')
        else lines="-"; p="(binary)"; fi
        echo "| $(basename "$f") | $size | $lines | ${p:-} |"
      done
}

if [ -n "$out" ]; then gen > "$out"; echo "wrote $out" >&2; else gen; fi
