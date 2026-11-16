base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1 and base.count("        __v0 = 1;") == 1
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
PAD4 = ("        AscendC::DataCopyExtParams __lp{1, 4, 0, 0, 0};\n"
        "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
        "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);")
# One arm per destination cell. The kernel reads exactly one, so mapping the block
# costs one build per cell.
for cell in (1, 2, 4, 7):
    src = base.replace(N, POISON + PAD4).replace("        __v0 = 1;", "        __v0 = %d;" % cell)
    open("probe-b30/e89_c%d.cpp" % cell, "w").write(src)
print("generated e89_c1 c2 c4 c7")
