# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conservative extern include scanning without invoking a compiler."""

import pytest
from pypto.runtime._extern_includes import UnsupportedArtifactInput, literal_includes


@pytest.mark.parametrize(
    "source, expected",
    [
        ('/*\n#include MACRO\n*/\n#include "real.h" // trailing comment', ["real.h"]),
        ('#if 0\n#if UNKNOWN\n#include BAD\n#endif\n#else\n#include "real.h"\n#endif', ["real.h"]),
        ('#if 1\n#include "real.h"\n#elif UNKNOWN\n#include BAD\n#else\n#include BAD\n#endif', ["real.h"]),
        ('#if UNKNOWN\n#include "a.h"\n#elif 1\n#include "b.h"\n#else\n#include BAD\n#endif', ["a.h", "b.h"]),
        ('#if (0)\n#include BAD\n#elif 0\n#include BAD\n#elif 1\n#include "real.h"\n#endif', ["real.h"]),
        ('const char* s = R"tag(\n#include MACRO\n)tag";\n#include "real.h"', ["real.h"]),
        ('// ignore \\\n#include MACRO\n#include "real.h"', ["real.h"]),
        ('#include \\\n"real.h"', ["real.h"]),
    ],
)
def test_only_possible_literal_includes_are_packaged(source, expected):
    assert list(literal_includes(source.encode())) == [('"', name) for name in expected]


@pytest.mark.parametrize("source", ["#include MACRO", "#if UNKNOWN\n#include MACRO\n#endif"])
def test_active_macro_include_has_specific_fallback_signal(source):
    with pytest.raises(UnsupportedArtifactInput, match="literal"):
        list(literal_includes(source.encode()))


@pytest.mark.parametrize("source", ["#endif", "#if 0", "#if 0\n#else\n#else\n#endif"])
def test_malformed_conditionals_are_not_classified_as_supported_fallback(source):
    with pytest.raises(ValueError) as exc:
        list(literal_includes(source.encode()))
    assert not isinstance(exc.value, UnsupportedArtifactInput)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
