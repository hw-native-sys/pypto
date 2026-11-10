# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Local process owner for one prepared static direct-MPI session."""

import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(filename: str) -> int:
    """Launch using a private local bootstrap file, never source bytes in argv."""
    from simpler.mpi_direct_bootstrap import decode_startup  # noqa: PLC0415
    from simpler.mpi_direct_supervisor import run_supervisor  # noqa: PLC0415

    path = Path(filename)
    settings = decode_startup(path.read_bytes())
    reports = {}
    try:
        return run_supervisor(
            settings["topology_path"],
            settings["controller"],
            mpirun_path=settings["mpirun_path"],
            launcher_family="auto",
            python_executable=settings["python_executable"],
            cleanup_reports=reports,
            startup_by_rank={int(rank): record for rank, record in settings["startup_by_rank"].items()},
        )
    finally:
        target = path.parent / "cleanup.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({str(rank): asdict(report) for rank, report in reports.items()}))
        temporary.replace(target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
