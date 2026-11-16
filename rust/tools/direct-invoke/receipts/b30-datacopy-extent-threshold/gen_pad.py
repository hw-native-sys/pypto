base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1
# Read element 0 -- the INDEX -- so the oracle's own PASS/FAIL is the verdict,
# not a decoded residue. The e1 probe set __v0 = 1; put it back to 0.
assert base.count("        __v0 = 1;") == 1
base = base.replace("        __v0 = 1;", "        __v0 = 0;")
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
for n in (1, 8):
    pad = ("        AscendC::DataCopyExtParams __lp{1, %d, 0, 0, 0};\n"
           "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
           "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);" % (n * 4))
    open("probe-b30/pad_%d.cpp" % n, "w").write(base.replace(N, POISON + pad))
# NEGATIVE CONTROL: same poison, same element-0 read, plain DataCopy at 4 bytes.
plain = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (1));"
open("probe-b30/padctl_1.cpp", "w").write(base.replace(N, POISON + plain))
print("generated pad_1, pad_8, padctl_1")
