// Kweli Lens AI front-end behaviour (no external libraries; loaded on every page).
(function () {
  "use strict";

  // ---- Input form: show only the panel for the selected input type.
  var form = document.getElementById("check-form");
  if (form) {
    var radios = form.querySelectorAll('input[name="input_type"]');
    var panels = form.querySelectorAll(".panel");
    var update = function () {
      var selected = form.querySelector('input[name="input_type"]:checked');
      var value = selected ? selected.value : "text";
      panels.forEach(function (p) {
        var active = p.getAttribute("data-panel") === value;
        p.classList.toggle("hidden", !active);
        // Disable inputs in hidden panels so they are not submitted.
        p.querySelectorAll("input, textarea").forEach(function (el) { el.disabled = !active; });
      });
    };
    radios.forEach(function (r) { r.addEventListener("change", update); });
    update();
    form.addEventListener("submit", function () {
      var btn = document.getElementById("submit-btn");
      if (btn) { btn.disabled = true; btn.textContent = "Uploading and starting..."; }
    });
  }

  // ---- Progress page: poll the job status until the report is ready.
  var progress = document.getElementById("progress");
  if (progress) {
    var statusUrl = progress.getAttribute("data-status-url");
    var reportUrl = progress.getAttribute("data-report-url");
    var msg = document.getElementById("progress-message");
    var errBox = document.getElementById("progress-error");
    var order = ["input", "claims", "checkworthy", "search", "verify", "verdict"];
    var markStage = function (stage) {
      var idx = order.indexOf(stage);
      progress.querySelectorAll(".stages li").forEach(function (li) {
        var i = order.indexOf(li.getAttribute("data-stage"));
        li.classList.toggle("done", idx > -1 && i < idx);
        li.classList.toggle("current", i === idx);
      });
    };
    var failures = 0;
    var showError = function (text) {
      errBox.textContent = text;
      errBox.classList.remove("hidden");
      progress.classList.add("failed");
    };
    var poll = function () {
      fetch(statusUrl, { credentials: "same-origin", cache: "no-store" })
        .then(function (r) {
          // 404: the job no longer exists (server restarted, report expired, or another browser) - stop polling.
          if (r.status === 404) { throw new Error("gone"); }
          if (!r.ok) { throw new Error("HTTP " + r.status); }
          return r.json();
        })
        .then(function (s) {
          failures = 0;
          if (s.state === "done") { window.location.href = reportUrl; return; }
          if (s.state === "error") { showError("The check failed: " + s.error); return; }
          msg.textContent = s.message || "Working...";
          markStage(s.stage);
          setTimeout(poll, 1500);
        })
        .catch(function (e) {
          if (e && e.message === "gone") {
            showError("This check is no longer available (the server may have been restarted). Please submit it again.");
            return;
          }
          failures += 1;
          if (failures >= 15) {  // about a minute without reaching the server
            showError("Lost contact with the Kweli Lens server. Check that it is still running, then refresh this page.");
            return;
          }
          setTimeout(poll, 4000);
        });
    };
    poll();
  }
})();
