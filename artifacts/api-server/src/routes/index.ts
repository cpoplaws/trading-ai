import { Router, type IRouter } from "express";
import healthRouter from "./health";
import stagingTradingRouter from "./staging-trading";

const router: IRouter = Router();

router.use(healthRouter);

/**
 * The staging mirror of the Python trading API exists purely so automated
 * mobile checks never hit the live trading engine. It is disabled in
 * production unless explicitly opted in.
 */
const stagingTradingEnabled =
  process.env["ENABLE_STAGING_TRADING_API"] === "1" ||
  process.env["NODE_ENV"] !== "production";

if (stagingTradingEnabled) {
  router.use("/staging-trading", stagingTradingRouter);
}

export default router;
