import sys
base = open("probe-b30/emb_e1_8.cpp").read()
NEEDLE = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(NEEDLE) == 1, "needle not unique"
POISON = ("        AscendC::Duplicate(buf0, 1.0e30f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
for n in (8, 4, 2, 1):
    copy = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (%d));" % n
    # variant A: poison prologue only. Nothing else about the kernel changes.
    open("probe-b30/pA_%d.cpp" % n, "w").write(base.replace(NEEDLE, POISON + copy))
    # variant B: poison prologue AND an MTE2->S barrier before the GetValue.
    open("probe-b30/pB_%d.cpp" % n, "w").write(
        base.replace(NEEDLE, POISON + copy + "\n        AscendC::PipeBarrier<PIPE_ALL>();"))
print("generated 8 sources")
