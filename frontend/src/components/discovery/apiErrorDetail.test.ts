import { describe, expect, it } from "vitest";
import { formatApiErrorDetail } from "@/components/discovery/apiErrorDetail";

/** Build an error shaped like the one `customInstance` throws. */
function apiError(status: number, body: unknown): Error {
  const err = new Error(`API error: ${status}`) as Error & {
    status: number;
    body: unknown;
  };
  err.status = status;
  err.body = body;
  return err;
}

describe("formatApiErrorDetail", () => {
  it("returns a string detail verbatim", () => {
    expect(formatApiErrorDetail(apiError(422, { detail: "bad discovery config" }))).toBe(
      "bad discovery config",
    );
  });

  it("returns a 500 string detail", () => {
    expect(formatApiErrorDetail(apiError(500, { detail: "malformed command" }))).toBe(
      "malformed command",
    );
  });

  it("joins pydantic items, prefixing each with its last loc element", () => {
    expect(
      formatApiErrorDetail(
        apiError(422, {
          detail: [
            {
              loc: ["body", "population"],
              msg: "Input should be greater than or equal to 2",
              type: "greater_than_equal",
            },
            {
              loc: ["body", "train_test_split"],
              msg: "Input should be less than 1",
              type: "less_than",
            },
          ],
        }),
      ),
    ).toBe(
      "population: Input should be greater than or equal to 2; " +
        "train_test_split: Input should be less than 1",
    );
  });

  it("uses the bare msg when a pydantic item has no loc", () => {
    expect(formatApiErrorDetail(apiError(422, { detail: [{ msg: "value is invalid" }] }))).toBe(
      "value is invalid",
    );
  });

  it("ignores non-pydantic array entries", () => {
    expect(
      formatApiErrorDetail(
        apiError(422, {
          detail: [42, { loc: ["body", "num_seeds"], msg: "too small" }],
        }),
      ),
    ).toBe("num_seeds: too small");
  });

  it("returns empty string for a non-ApiError", () => {
    expect(formatApiErrorDetail(new Error("network down"))).toBe("");
    expect(formatApiErrorDetail(undefined)).toBe("");
    expect(formatApiErrorDetail("nope")).toBe("");
  });

  it("returns empty string when detail is missing, empty, or unusable", () => {
    expect(formatApiErrorDetail(apiError(422, {}))).toBe("");
    expect(formatApiErrorDetail(apiError(422, { detail: "" }))).toBe("");
    expect(formatApiErrorDetail(apiError(422, { detail: [] }))).toBe("");
    expect(formatApiErrorDetail(apiError(422, { detail: 7 }))).toBe("");
    expect(formatApiErrorDetail(apiError(422, undefined))).toBe("");
  });
});
