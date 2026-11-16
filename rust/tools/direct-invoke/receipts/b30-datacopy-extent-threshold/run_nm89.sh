set -u
cd /tmp/di-i27
python3 gen_nm89.py || exit 1
# NM 89 -- IS THE REPAIR ITSELF SAFE? A 4-byte DataCopyPad delivers element 0 (measured),
# but nothing has measured what it does to elements 1..7. `isPad = false` specifies no
# padding value and the header does not say what becomes of the block remainder. Three
# behaviours fit everything run so far: write 4 bytes and leave the rest; read-modify-write
# a whole 32-byte block and overwrite them from GM; or fill the remainder with zero/garbage.
# The emitter now applies this pad AUTOMATICALLY, so a block-wide write would be silent
# corruption of live data -- the same defect shape the repair exists to fix.
#
# Poison 2.0 across all 8 cells, 4-byte pad, then read ELEMENT 1:
#   got 2560.125 -> element 1 = 2.0  POISON PRESERVED: the pad wrote only its 4 bytes. SAFE.
#   got    0.125 -> element 1 = 0.0  the GM decoy ARRIVED: the pad wrote a whole block. RISK.
#   anything else -> neither; do not interpret without the stale-cell row.
# CONTROL: the 32-byte pad must read 0.0, since there all 8 elements are legitimately
# copied -- it proves the element-1 readout works under a pad at all.
#
# NOTE ON READING THIS: on a PASS the oracle prints no `got ... want` line, so both the
# verdict and the got value are printed below and the verdict is authoritative. These arms
# read element 1 as the index, so they FAIL by construction and `got` is the instrument.
for k in e1pad_1 e1pad_8; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/$k.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/nm89_${k}_rep${rep}.log" 2>&1
    rc=$?
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/nm89_${k}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/nm89_${k}_rep${rep}.log")
    case "$g" in
      "got 2560.125000") d="element1=2.0  POISON PRESERVED -> pad wrote only its bytes" ;;
      "got 0.125000")    d="element1=0.0  GM decoy arrived -> pad wrote a WHOLE BLOCK" ;;
      "got 640.125000")  d="element1=0.5  the stale cell, neither" ;;
      "")                d="no got line (rc=$rc, verdict ${v:-?})" ;;
      *)                 d="unexpected [$g]" ;;
    esac
    printf '  %-9s rep%s | %-5s | %s\n' "$k" "$rep" "${v:-?}" "$d"
  done
done
echo "--- faults ---"; grep -hoE "507[0-9]{3}" logs/nm89_*.log 2>/dev/null | sort | uniq -c
