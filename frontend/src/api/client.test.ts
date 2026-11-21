import { describe, expect, it } from "vitest";
import { type ApiError, apiErrorDetail } from "./client";

function apiError(body: unknown): ApiError {
  const err = new Error("API error: 422 Unprocessable Entity") as ApiError;
  err.status = 422;
  err.body = body;
  return err;
}

describe("apiErrorDetail", () => {
  it("returns a string detail as is", () => {
    expect(apiErrorDetail(apiError({ detail: "Strategy not found" }))).toBe("Strategy not found");
  });

  it("joins pydantic validation messages without the 'Value error, ' prefix", () => {
    const body = {
      detail: [
        {
          loc: ["body", "dsl_config"],
          msg: "Value error, ADAPTIVE_RSI alpha must be between 0.7 and 1.0 (got 0.3)",
          type: "value_error",
        },
        { loc: ["body", "name"], msg: "Field required", type: "missing" },
      ],
    };
    expect(apiErrorDetail(apiError(body))).toBe(
      "ADAPTIVE_RSI alpha must be between 0.7 and 1.0 (got 0.3); Field required",
    );
  });

  it("is undefined without a readable reason", () => {
    expect(apiErrorDetail(new Error("network down"))).toBeUndefined();
    expect(apiErrorDetail(apiError(undefined))).toBeUndefined();
    expect(apiErrorDetail(apiError({ detail: [{ loc: [] }] }))).toBeUndefined();
  });
});
