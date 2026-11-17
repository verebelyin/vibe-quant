import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type { DslConfig } from "./types";
import { parseYamlToDsl, serializeConfig, type YamlError } from "./yamlConfig";

interface YamlEditorProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
  /** Current parse error (null when the text parses), so the page can block Save. */
  onErrorChange?: ((error: string | null) => void) | undefined;
}

export function YamlEditor({ config, onConfigChange, onErrorChange }: YamlEditorProps) {
  const [text, setText] = useState(() => serializeConfig(config));
  const [error, setError] = useState<YamlError | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Monotonic counter: incremented on every local edit, checked on incoming sync.
  // If the incoming config's serialized YAML matches current text, skip the sync
  // (handles the echo-back case without fragile reference equality).
  const localEditCounter = useRef(0);
  const lastSyncedCounter = useRef(0);

  // Sync from visual editor -> YAML text when config changes externally
  useEffect(() => {
    if (localEditCounter.current !== lastSyncedCounter.current) {
      // We have a pending local edit — check if incoming config matches current text
      const incomingYaml = serializeConfig(config);
      if (incomingYaml === text) {
        // Echo-back of our own edit — skip
        lastSyncedCounter.current = localEditCounter.current;
        return;
      }
      // Content actually differs (visual editor changed something else) — accept it
      lastSyncedCounter.current = localEditCounter.current;
    }
    const newText = serializeConfig(config);
    if (newText === text) return; // no-op if already in sync
    setText(newText);
    setError(null);
    onErrorChange?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config, onErrorChange]);

  // The text lives here: once this editor unmounts, its parse error no longer applies.
  useEffect(() => () => onErrorChange?.(null), [onErrorChange]);

  const handleTextChange = useCallback(
    (value: string) => {
      localEditCounter.current += 1;
      setText(value);
      const result = parseYamlToDsl(value);
      setError(result.error);
      onErrorChange?.(result.error?.message ?? null);
      if (!result.error) {
        onConfigChange(result.config);
      }
    },
    [onConfigChange, onErrorChange],
  );

  const handleFormat = useCallback(() => {
    const result = parseYamlToDsl(text);
    if (result.error) {
      setError(result.error);
      onErrorChange?.(result.error.message);
      return;
    }
    const formatted = serializeConfig(result.config);
    setText(formatted);
    setError(null);
    onErrorChange?.(null);
  }, [text, onErrorChange]);

  const handleUpload = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const content = ev.target?.result;
        if (typeof content === "string") {
          handleTextChange(content);
        }
      };
      reader.readAsText(file);
      // Reset input so same file can be re-uploaded
      e.target.value = "";
    },
    [handleTextChange],
  );

  const handleDownload = useCallback(() => {
    const blob = new Blob([text], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "strategy.yaml";
    a.click();
    URL.revokeObjectURL(url);
  }, [text]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(
      () => toast.success("YAML copied to clipboard"),
      () => toast.error("Failed to copy to clipboard"),
    );
  }, [text]);

  const handlePaste = useCallback(() => {
    navigator.clipboard.readText().then(
      (clipText) => {
        if (clipText.trim()) {
          handleTextChange(clipText);
          toast.success("YAML pasted from clipboard");
        } else {
          toast.error("Clipboard is empty");
        }
      },
      () => toast.error("Failed to read clipboard"),
    );
  }, [handleTextChange]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm text-muted-foreground">
          Edit strategy config as YAML. Changes sync bidirectionally with the visual editor.
        </Label>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            Copy
          </Button>
          <Button variant="outline" size="sm" onClick={handlePaste}>
            Paste
          </Button>
          <Button variant="outline" size="sm" onClick={handleFormat}>
            Format
          </Button>
          <Button variant="outline" size="sm" onClick={handleUpload}>
            Upload
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            Download
          </Button>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml"
        className="hidden"
        onChange={handleFileChange}
      />

      <textarea
        value={text}
        onChange={(e) => handleTextChange(e.target.value)}
        spellCheck={false}
        className="min-h-[400px] w-full rounded-md border border-border bg-input p-3 font-mono text-sm leading-relaxed text-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring dark:bg-input/30"
      />

      {error && (
        <div className="rounded-md border border-destructive bg-destructive/10 px-3 py-2">
          <p className="text-sm text-destructive">
            {error.line != null && <span className="font-semibold">Line {error.line}: </span>}
            {error.message}
          </p>
        </div>
      )}
    </div>
  );
}
