import { useState } from "react";
import { useListSymbolsApiDataSymbolsGet } from "@/api/generated/data/data";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DataBrowser } from "./DataBrowser";
import { DataQualityPanel } from "./DataQualityPanel";

export function DataBrowserTab() {
  const [qualitySymbol, setQualitySymbol] = useState("");

  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const symbols = symbolsQuery.data?.data ?? [];

  return (
    <Tabs defaultValue="browser" className="space-y-4">
      <TabsList>
        <TabsTrigger value="browser">Browser</TabsTrigger>
        <TabsTrigger value="quality">Quality</TabsTrigger>
      </TabsList>

      <TabsContent value="browser">
        <DataBrowser />
      </TabsContent>

      <TabsContent value="quality">
        <div className="space-y-4">
          {/* Symbol selector for quality */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs uppercase tracking-wider text-muted-foreground">Symbol</Label>
            <Select value={qualitySymbol} onValueChange={setQualitySymbol}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Select symbol..." />
              </SelectTrigger>
              <SelectContent>
                {symbols.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {!qualitySymbol && (
            <div className="rounded-lg border p-8 text-center text-sm text-muted-foreground">
              Select a symbol to view quality metrics
            </div>
          )}

          {qualitySymbol && <DataQualityPanel symbol={qualitySymbol} />}
        </div>
      </TabsContent>
    </Tabs>
  );
}
