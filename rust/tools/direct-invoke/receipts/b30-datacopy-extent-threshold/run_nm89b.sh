set -u
cd /tmp/di-i27
python3 gen_nm89b.py || exit 1
# NM 89 FOLLOW-UP: HOW FAR past its four bytes does the pad write?
# Poison 2.0 across all 8 cells, then a 4-byte DataCopyPad, then read cell 1/2/4/7.
# GM holds [3.0, 0.0, 0.0, ...]; got = 1280*value + 0.125, so:
#   got 2560.125 -> 2.0  POISON PRESERVED: the pad did not touch this cell
#   got 3840.125 -> 3.0  GM[0] REPLICATED into this cell  (verdict PASS)
#   got    0.125 -> 0.0  GM[cell] arrived: a whole-block copy from GM
# Cell 1 already read 3.0 three times of three. If 2/4/7 also read 3.0 the single
# element is broadcast across the destination block; if they hold 2.0 the write is
# bounded near the start; if they hold 0.0 it is a block copy after all.
for cell in 1 2 4 7; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/e89_c$cell.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/nm89b_c${cell}_rep${rep}.log" 2>&1
    rc=$?
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/nm89b_c${cell}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/nm89b_c${cell}_rep${rep}.log")
    case "$g" in
      "got 2560.125000") d="2.0  POISON PRESERVED (untouched)" ;;
      "got 3840.125000") d="3.0  GM[0] REPLICATED here" ;;
      "got 0.125000")    d="0.0  GM[cell] arrived (block copy)" ;;
      "got 640.125000")  d="0.5  stale cell" ;;
      "")                d="no got line -> verdict ${v:-?} means value 3.0 (PASS => index 3)" ;;
      *)                 d="unexpected [$g]" ;;
    esac
    printf '  cell %-2s rep%s | %-5s | %s\n' "$cell" "$rep" "${v:-?}" "$d"
  done
done
echo "--- faults ---"; grep -hoE "507[0-9]{3}" logs/nm89b_*.log 2>/dev/null | sort | uniq -c
