import sys
base = open("probe-b30/emb_e1_8.cpp").read()
NEEDLE = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(NEEDLE) == 1
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
for n in (8, 7, 6, 5):
    copy = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (%d));" % n
    open("probe-b30/pA_%d.cpp" % n, "w").write(base.replace(NEEDLE, POISON + copy))
print("generated 4 sources for 5,6,7,8 f32")
