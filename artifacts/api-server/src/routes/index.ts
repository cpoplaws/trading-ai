import { Router, type IRouter } from "express";
import healthRouter from "./health";
import stagingTradingRouter, {
  isStagingTradingEnabled,
} from "./staging-trading";

const router: IRouter = Router();

router.use(healthRouter);

/**
 * The staging mirror of the Python trading API exists purely so automated
 * mobile and dashboard checks never hit the live trading engine. It is
 * disabled in production unless explicitly opted in.
 */
if (isStagingTradingEnabled()) {
  router.use("/staging-trading", stagingTradingRouter);
}

export default router;
