function getToken() {
  return localStorage.getItem("access_token");
}

function authHeaders(extra = {}) {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function apiFetch(url, options = {}) {
  const headers = authHeaders(options.headers || {});
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed with ${response.status}`);
  }
  return response.json();
}

function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

function badge(status) {
  return `<span class="badge ${status}">${status.replaceAll("_", " ")}</span>`;
}

async function initLogin() {
  const form = document.getElementById("loginForm");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setText("loginStatus", "Signing in...");
    try {
      const payload = Object.fromEntries(new FormData(form).entries());
      const result = await apiFetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      localStorage.setItem("access_token", result.access_token);
      setText("loginStatus", "Sign in successful. Redirecting...");
      window.location.href = "/";
    } catch (error) {
      setText("loginStatus", error.message);
    }
  });
}

async function initDashboard() {
  const cards = document.getElementById("summaryCards");
  if (!cards) return;
  try {
    const summary = await apiFetch("/api/dashboard/summary");
    cards.innerHTML = [
      ["Total Candidates", summary.total_candidates],
      ["Shortlisted", `${summary.shortlisted_count} (${summary.shortlisted_percentage}%)`],
      ["Scheduled", `${summary.scheduled_count} (${summary.scheduled_percentage}%)`],
      ["Conversion", `${summary.conversion_rate}%`],
    ].map(([label, value]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong></article>`).join("");
    setText("workflowStatus", `Drop-offs tracked: ${summary.dropoff_count}`);
  } catch (error) {
    cards.innerHTML = `<article class="metric-card">${error.message}</article>`;
    setText("workflowStatus", "Unable to load analytics");
  }
}

async function initUpload() {
  const form = document.getElementById("uploadForm");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setText("uploadStatus", "Uploading resume and queueing analysis...");
    try {
      const formData = new FormData(form);
      const result = await apiFetch("/api/candidates/upload", { method: "POST", body: formData });
      setText("uploadStatus", `Candidate ${result.candidate_id} queued for workflow processing`);
      form.reset();
    } catch (error) {
      setText("uploadStatus", error.message);
    }
  });
}

async function initCandidateDetail() {
  const container = document.getElementById("candidateDetail");
  if (!container) return;
  const candidateId = container.dataset.candidateId;
  try {
    const candidate = await apiFetch(`/api/candidates/${candidateId}`);
    const score = candidate.latest_score || {};
    container.innerHTML = `
      <article class="panel">
        <h3>${candidate.full_name}</h3>
        <div class="kv">
          <div class="row"><span>Email</span><span>${candidate.email}</span></div>
          <div class="row"><span>Phone</span><span>${candidate.phone || "-"}</span></div>
          <div class="row"><span>Final Score</span><span>${candidate.final_score || "-"}</span></div>
        </div>
      </article>
      <article class="panel">
        <h3>Scoring</h3>
        <div class="kv">
          <div class="row"><span>ATS Score</span><span>${score.ats_score || "-"}</span></div>
          <div class="row"><span>Embedding Score</span><span>${score.embedding_score || "-"}</span></div>
          <div class="row"><span>Match %</span><span>${score.match_percentage || "-"}</span></div>
        </div>
      </article>
      <article class="panel">
        <h3>Resume Summary</h3>
        <p>${score.experience_summary || candidate.resume?.structured_data?.word_count || "No summary available"}</p>
      </article>
    `;
    document.getElementById("candidateStatusBadge").outerHTML = badge(candidate.status);
  } catch (error) {
    container.innerHTML = `<article class="panel">${error.message}</article>`;
  }
}

async function initDecisionPanel() {
  const form = document.getElementById("decisionForm");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setText("decisionStatus", "Submitting decision...");
    try {
      const payload = Object.fromEntries(new FormData(form).entries());
      const candidateId = payload.candidate_id;
      delete payload.candidate_id;
      const result = await apiFetch(`/api/candidates/${candidateId}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setText("decisionStatus", `Decision saved: ${result.decision}`);
    } catch (error) {
      setText("decisionStatus", error.message);
    }
  });
}

async function initInterviewStatus() {
  const container = document.getElementById("interviewTable");
  if (!container) return;
  try {
    const items = await apiFetch("/api/interviews/status");
    if (!items.length) {
      container.textContent = "No interviews scheduled yet.";
      return;
    }
    container.innerHTML = `
      <table class="table">
        <thead>
          <tr><th>ID</th><th>Candidate</th><th>Round</th><th>Status</th><th>Meeting</th></tr>
        </thead>
        <tbody>
          ${items.map((item) => `
            <tr>
              <td>${item.id}</td>
              <td>${item.candidate_id}</td>
              <td>${item.round_number}</td>
              <td>${badge(item.status)}</td>
              <td>${item.meeting_url ? `<a href="${item.meeting_url}" target="_blank">Open</a>` : "-"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } catch (error) {
    container.textContent = error.message;
  }
}

async function initAnalytics() {
  const cards = document.getElementById("analyticsCards");
  if (!cards) return;
  try {
    const overview = await apiFetch("/api/analytics/overview");
    cards.innerHTML = Object.entries(overview.by_status)
      .map(([label, value]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong></article>`)
      .join("");
    document.getElementById("funnelList").innerHTML = overview.funnel
      .map((point) => `<div class="row"><span>${point.label}</span><strong>${point.value}</strong></div>`)
      .join("");
  } catch (error) {
    cards.innerHTML = `<article class="metric-card">${error.message}</article>`;
  }
}

function initLogout() {
  const button = document.getElementById("logoutButton");
  if (!button) return;
  button.addEventListener("click", () => {
    localStorage.removeItem("access_token");
    window.location.href = "/login";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initLogout();
  initLogin();
  initDashboard();
  initUpload();
  initCandidateDetail();
  initDecisionPanel();
  initInterviewStatus();
  initAnalytics();
});

