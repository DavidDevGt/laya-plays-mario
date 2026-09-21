// Drives the panel purely from episode.jsonl (one line per real decision, produced by
// record_demo.py while it actually ran the checkpoint) and the video's own playback clock.
// No value here is computed independently of that file -- this script only looks up "which
// logged decision was current at this video timestamp" and renders its fields verbatim.

const video = document.getElementById("video");
let events = [];
let meta = null;
let lastRenderedIndex = -1;

async function load() {
  const [metaRes, jsonlRes] = await Promise.all([
    fetch("/episode_meta.json"),
    fetch("/episode.jsonl"),
  ]);
  meta = await metaRes.json();
  const text = await jsonlRes.text();
  events = text.trim().split("\n").map((line) => JSON.parse(line));

  document.getElementById("checkpoint-name").textContent = meta.checkpoint.split("/").pop();
  document.getElementById("cfg-threshold").textContent = meta.threshold;
  document.getElementById("cfg-hold").textContent = meta.hold_decisions;
  document.getElementById("cfg-stuck").textContent = meta.stuck_threshold;

  const badge = document.getElementById("outcome-badge");
  badge.textContent = `${meta.outcome} · distance ${meta.final_distance} · time_used ${meta.final_time_used}/400`;
  if (meta.outcome === "FLAG") badge.classList.add("flag");

  video.src = "/episode.mp4";
  video.play().catch(() => {});
  requestAnimationFrame(tick);
}

function findEventAt(t) {
  // events are ordered by timestamp_s; binary search for the last one <= t
  let lo = 0, hi = events.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (events[mid].timestamp_s <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans;
}

function fmt(v, digits = 0) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) && digits === 0 ? v : v.toFixed(digits);
  if (typeof v === "boolean") return v ? "true" : "false";
  return v;
}

function render(e) {
  // --- jump probability bars ---
  const isFresh = e.decision_source === "model";
  const p = isFresh ? e.jump_probability : e.last_real_jump_probability;
  const pctYes = p !== null && p !== undefined ? Math.round(p * 100) : 0;
  const pctNo = 100 - pctYes;
  document.getElementById("bar-yes").style.width = pctYes + "%";
  document.getElementById("bar-no").style.width = pctNo + "%";
  document.getElementById("p-yes").textContent = p !== null && p !== undefined ? p.toFixed(3) : "–";
  document.getElementById("p-no").textContent = p !== null && p !== undefined ? (1 - p).toFixed(3) : "–";

  const srcEl = document.getElementById("decision-source");
  srcEl.className = "decision-source " + e.decision_source;
  if (e.decision_source === "model") {
    srcEl.textContent = "fresh model inference this decision";
  } else if (e.decision_source === "held") {
    srcEl.textContent = `holding a committed jump (controller hysteresis) — last real P(jump)=${(e.last_real_jump_probability ?? 0).toFixed(3)} at decision #${e.last_real_decision_index}`;
  } else {
    srcEl.textContent = `stuck-against-wall safety fallback forced this jump — no model call this decision`;
  }

  const confEl = document.getElementById("confidence-val");
  if (p !== null && p !== undefined) {
    confEl.textContent = (Math.abs(p - 0.5) * 2).toFixed(3);
  } else {
    confEl.textContent = "–";
  }

  // --- selected action + gamepad ---
  const actionText = e.selected_jump ? "RIGHT (fixed) + A (jump)" : "RIGHT (fixed)";
  document.getElementById("selected-action").textContent = actionText;
  document.getElementById("button-list").textContent = e.buttons.join(" + ");

  document.getElementById("btn-right").classList.add("active"); // move is always RIGHT in this config
  document.getElementById("btn-left").classList.remove("active");
  document.getElementById("btn-A").classList.toggle("active", e.buttons.includes("A"));
  document.getElementById("btn-B").classList.toggle("active", e.buttons.includes("B"));

  document.getElementById("hold-info").textContent =
    e.hold_frames_remaining > 0 ? `hold_frames_remaining: ${e.hold_frames_remaining}` : "";

  // --- game state ---
  document.getElementById("s-x").textContent = fmt(e.mario_x);
  document.getElementById("s-y").textContent = fmt(e.mario_y);
  document.getElementById("s-vx").textContent = fmt(e.velocity_x);
  document.getElementById("s-vy").textContent = fmt(e.velocity_y);
  document.getElementById("s-ground").textContent = fmt(e.on_ground);
  document.getElementById("s-gap").textContent = fmt(e.gap_ahead);
  document.getElementById("s-enemy-dx").textContent = e.nearest_enemy_dx === null ? "none visible" : e.nearest_enemy_dx;
  document.getElementById("s-enemy-dy").textContent = e.nearest_enemy_dy === null ? "–" : e.nearest_enemy_dy;
  document.getElementById("s-enemy-type").textContent = e.nearest_enemy_type === null ? "–" : e.nearest_enemy_type;
  document.getElementById("s-enemies").textContent = fmt(e.num_enemies_onscreen);

  // --- performance ---
  document.getElementById("m-time").textContent = `${e.game_time_used}/400`;
  document.getElementById("m-distance").textContent = e.mario_x;
  document.getElementById("m-decision").textContent = `${e.decision_index + 1} / ${events.length}`;
  document.getElementById("m-latency").textContent = e.model_latency_ms !== null ? `${e.model_latency_ms.toFixed(2)} ms` : "– (no model call)";
}

function tick() {
  if (events.length && !video.paused) {
    const idx = findEventAt(video.currentTime);
    if (idx !== lastRenderedIndex) {
      lastRenderedIndex = idx;
      render(events[idx]);
    }
  }
  requestAnimationFrame(tick);
}

// also render on manual scrub while paused
video.addEventListener("timeupdate", () => {
  if (events.length) {
    const idx = findEventAt(video.currentTime);
    if (idx !== lastRenderedIndex) {
      lastRenderedIndex = idx;
      render(events[idx]);
    }
  }
});

load();
