base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1
# Element 1 is the cell under test, so __v0 STAYS 1 -- the e1 base already has it.
assert base.count("        __v0 = 1;") == 1, "base must read element 1"
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
for n in (1, 8):
    pad = ("        AscendC::DataCopyExtParams __lp{1, %d, 0, 0, 0};\n"
           "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
           "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);" % (n * 4))
    open("probe-b30/e1pad_%d.cpp" % n, "w").write(base.replace(N, POISON + pad))
print("generated e1pad_1 (4 B pad) and e1pad_8 (32 B pad control)")
