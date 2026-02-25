import yaml from "js-yaml";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type { DslConfig } from "./types";
import { parseDslConfig } from "./types";

interface YamlEditorProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

interface YamlError {
  message: string;
  line?: number | undefined;
}

function serializeConfig(config: DslConfig): string {
  return yaml.dump(config, { indent: 2, lineWidth: 120, noRefs: true });
}

function parseYamlToDsl(text: string): { config: DslConfig; error: YamlError | null } {
  try {
    const parsed = yaml.load(text);
    if (parsed === null || parsed === undefined || typeof parsed !== "object") {
      return {
        config: {} as DslConfig,
        error: { message: "YAML must be a mapping (object), not a scalar or array" },
      };
    }
    const config = parseDslConfig(parsed as Record<string, unknown>);
    return { config, error: null };
  } catch (e: unknown) {
    if (e instanceof yaml.YAMLException) {
      return {
        config: {} as DslConfig,
        error: {
          message: e.message,
          line: e.mark?.line !== undefined ? e.mark.line + 1 : undefined,
        },
      };
    }
    const msg = e instanceof Error ? e.message : "Unknown parse error";
    return { config: {} as DslConfig, error: { message: msg } };
  }
}

export function YamlEditor({ config, onConfigChange }: YamlEditorProps) {
  const [text, setText] = useState(() => serializeConfig(config));
  const [error, setError] = useState<YamlError | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Track whether config was updated externally (visual editor)
  const skipNextSync = useRef(false);

  // Sync from visual editor -> YAML text when config changes externally
  useEffect(() => {
    if (skipNextSync.current) {
      skipNextSync.current = false;
      return;
    }
    setText(serializeConfig(config));
    setError(null);
  }, [config]);

  const handleTextChange = useCallback(
    (value: string) => {
      setText(value);
      const result = parseYamlToDsl(value);
      setError(result.error);
      if (!result.error) {
        skipNextSync.current = true;
        onConfigChange(result.config);
      }
    },
    [onConfigChange],
  );

  const handleFormat = useCallback(() => {
    const result = parseYamlToDsl(text);
    if (result.error) {
      setError(result.error);
      return;
    }
    const formatted = serializeConfig(result.config);
    setText(formatted);
    setError(null);
  }, [text]);

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
