import { Badge } from "@/components/ui/badge";

export function PassBadge({ passed }: { passed: boolean | null | undefined }) {
  if (passed == null) return <Badge variant="secondary">N/A</Badge>;
  if (passed) return <Badge>PASS</Badge>;
  return <Badge variant="destructive">FAIL</Badge>;
}
