import {describe, expect, it, vi} from "vitest";
import {api, ApiError} from "./api";

const response = (body: string, status = 200, type = "application/json") => new Response(body, {status, headers: {"content-type": type}});
describe("API response validation", () => {
  it("explains an HTML Vite fallback response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response("<script type=module>", 200, "text/html")));
    await expect(api("/overview")).rejects.toThrow(/Vite.*proxy/i);
  });
  it("explains a legitimate unavailable genuine dataset", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response('{"detail":"No verified run"}', 503)));
    await expect(api("/overview")).rejects.toMatchObject({status: 503} satisfies Partial<ApiError>);
  });
  it("rejects malformed non-JSON responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response("not json")));
    await expect(api("/overview")).rejects.toThrow(/invalid non-JSON/i);
  });
});
