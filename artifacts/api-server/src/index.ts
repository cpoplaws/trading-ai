import { createServer } from "node:http";
import app from "./app";
import { logger } from "./lib/logger";
import {
  attachStagingTradingSocket,
  isStagingTradingEnabled,
} from "./routes/staging-trading";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// An explicit http server (rather than `app.listen`) so the staging trading
// mirror can attach its Socket.IO live-update channel to the same port.
const httpServer = createServer(app);

if (isStagingTradingEnabled()) {
  attachStagingTradingSocket(httpServer);
}

httpServer.on("error", (err) => {
  logger.error({ err }, "Error listening on port");
  process.exit(1);
});

httpServer.listen(port, () => {
  logger.info({ port }, "Server listening");
});
