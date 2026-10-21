/* Added by the local review server before the existing review UI scripts. */
(() => {
  "use strict";
  const originalFetch = window.fetch.bind(window);
  let session;
  async function token() {
    if (!session) {
      session = originalFetch("/api/review-session", { cache: "no-store", credentials: "same-origin" })
        .then(async response => {
          if (!response.ok) throw new Error("Review session unavailable; reload the review page.");
          return (await response.json()).token;
        }).catch(error => { session = null; throw error; });
    }
    return session;
  }
  async function reviewFetch(input, options = {}) {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url, location.href);
    const method = String(options.method || input.method || "GET").toUpperCase();
    if (url.origin !== location.origin || !["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
      return originalFetch(input, options);
    }
    const headers = new Headers(options.headers || input.headers || {});
    headers.set("X-Review-Token", await token());
    const response = await originalFetch(input, { ...options, headers, credentials: "same-origin" });
    if (response.status === 403) session = null;
    return response;
  }
  window.reviewFetch = reviewFetch;
  // Historical STT HTML is kept in the user's data workspace rather than this repo.
  // Protect its existing same-origin fetch calls without changing saved review data.
  window.fetch = reviewFetch;
})();
