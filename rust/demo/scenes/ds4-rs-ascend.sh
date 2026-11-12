# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Scene 5 of 5: ds4-rs-ascend -- planning and serving a model on Ascend.
#
# NOT everything here works without a card, and the scene says so rather than
# working around it. Measured on a machine with no Ascend driver: `version`,
# `help`, `help --advanced` and the kernel registry all work, while `mem` and
# `plan` fail because they shell out to `npu-smi`.
#
# That is worth showing rather than hiding. The crate's own manifest comment
# claims the planning half works without the `ascend` cargo feature, and it
# does -- but the feature is not the only dependency: the memory profile needs
# the driver on PATH either way, and `--mem-source` offers `npu-smi` or `acl`
# and nothing synthetic. So a capacity question still needs a box.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
ROOT="$(demo_root)"; cd "$ROOT"
DS4="$(tool ds4-rs-ascend)"

beat "ds4-rs-ascend -- deploying DeepSeek V4.1 Flash"
note "A serving utility. Memory profile, parallel split, kernel-set registry."
run "$DS4" version
note "A version line that identifies the build it came from."
pause 6

beat "The command surface"
run "$DS4" help
pause 12

beat "Two tiers of help, by design"
note "The everyday options above; the expert knobs only when asked for."
note "An advanced flag still WORKS without this -- it only governs whether it is documented,"
note "so a support engineer can direct someone to one without it being noise for everyone."
run "$DS4" help --advanced
pause 14

beat "What needs a card, and what the tool says when there is none"
note "This machine has no Ascend driver. Watch how the memory profile reports that."
run "$DS4" mem
expect_refusal "it names the missing driver and the command it tried, not a stack trace"
note "And planning needs the same profile, so it stops at the same place."
run "$DS4" plan --mode latency
expect_refusal "a capacity question needs a box: --mem-source offers npu-smi or acl, nothing synthetic"
note "Worth being exact about this. The crate's manifest says the planning half works"
note "without the ascend cargo feature, and that is true -- but the feature was never the"
note "only dependency. The driver has to be on PATH either way."
pause 10

beat "What the split would be chosen from"
note "--mode is the deployment INTENT, and the split follows rather than being picked:"
note "latency for one user; throughput for aggregate tokens per second;"
note "balanced for a middle point; max-context for the longest context the box can hold."
note "TP, DP, EP and PP are not a free choice -- two do nothing for a single stream,"
note "and two cost memory linearly, so the configuration follows from the objective."
pause 10

beat "And what this tool does NOT claim, in its own words"
note "The parallel module's own header says it: nothing in it has run on multiple chips."
note "The collective cost is an assumption, and so is the claim that n host links"
note "deliver n times one link's bandwidth. That is the first thing to measure."
note "Saying so is part of the tool rather than a footnote somewhere else."
pause 10

beat "The kernel-set registry, which does work here"
note "Kernel sets are content-addressed, hashed, and their index is signature-verified."
run "$DS4" kernels list
note "Empty on a fresh box, and it says how to get some rather than just reporting nothing."
pause 8

beat "The index is signature-verified"
note "Each set is content-addressed and hashed, and the index that names them is signed."
note "The check answers one question: are these the kernels that were published."
run "$DS4" kernels verify
pause 10

beat "Tuning for this box"
note "profile sweeps launch and tiling knobs on this silicon and writes a tuning file"
note "the deploy step then loads. The description is always available, so the"
note "capability stays discoverable before anyone spends a box on it."
run "$DS4" profile --model ds4 --mode latency --out rust/ds4-tune.json
pause 12

demo_end
