import { DatabaseTab } from "@/components/settings/DatabaseTab";
import { LatencyTab } from "@/components/settings/LatencyTab";
import { RiskTab } from "@/components/settings/RiskTab";
import { SizingTab } from "@/components/settings/SizingTab";
import { SystemTab } from "@/components/settings/SystemTab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function SettingsPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-foreground">Settings</h1>

      <Tabs defaultValue="sizing" className="mt-4">
        <TabsList variant="line">
          <TabsTrigger value="sizing">Sizing</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="latency">Latency</TabsTrigger>
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="sizing" className="mt-6">
          <SizingTab />
        </TabsContent>
        <TabsContent value="risk" className="mt-6">
          <RiskTab />
        </TabsContent>
        <TabsContent value="latency" className="mt-6">
          <LatencyTab />
        </TabsContent>
        <TabsContent value="database" className="mt-6">
          <DatabaseTab />
        </TabsContent>
        <TabsContent value="system" className="mt-6">
          <SystemTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
