import { ActiveJobsPanel } from "@/components/backtest/ActiveJobsPanel";
import { BacktestLaunchForm } from "@/components/backtest/BacktestLaunchForm";

export function BacktestPage() {
  return (
    <div className="flex flex-col gap-6">
      <BacktestLaunchForm />
      <ActiveJobsPanel />
    </div>
  );
}
