export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) { super(message); }
}

/** Parse only JSON API responses and explain the common Vite fallback failure. */
export async function api<T>(path: string): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`); }
  catch { throw new ApiError("Cannot reach the API. Start FastAPI on 127.0.0.1:8000 and the Vite dev server from dashboard/."); }
  const contentType = response.headers.get("content-type") || "";
  const body = await response.text();
  if (contentType.includes("text/html") || body.trimStart().startsWith("<")) {
    throw new ApiError("The dashboard received HTML instead of API JSON. Check that Vite's /api proxy targets FastAPI on http://127.0.0.1:8000.", response.status);
  }
  let payload: unknown;
  try { payload = body ? JSON.parse(body) : {}; }
  catch { throw new ApiError("The API returned an invalid non-JSON response. Verify the backend is running and compatible.", response.status); }
  if (!response.ok) {
    const detail = typeof payload === "object" && payload && "detail" in payload ? String((payload as {detail: unknown}).detail) : "Request failed";
    if (response.status === 503) throw new ApiError(`Verified dataset unavailable: ${detail} Load a genuine verified snapshot or run the pipeline.`, 503);
    throw new ApiError(detail, response.status);
  }
  return payload as T;
}
