// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

import net from "node:net";

const targetHost = process.env.RELAY_TARGET_HOST;
const targetPort = Number.parseInt(process.env.RELAY_TARGET_PORT ?? "", 10);
const listenPort = 7895;

if (!targetHost || !Number.isInteger(targetPort) || targetPort < 1 || targetPort > 65535) {
  throw new Error("RELAY_TARGET_HOST and RELAY_TARGET_PORT must identify the trusted host relay");
}

const server = net.createServer((client) => {
  const upstream = net.createConnection({ host: targetHost, port: targetPort });

  client.on("error", () => upstream.destroy());
  upstream.on("error", () => client.destroy());
  client.pipe(upstream);
  upstream.pipe(client);
});

server.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
});

server.listen(listenPort, "0.0.0.0");

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
