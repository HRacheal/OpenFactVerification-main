// Kweli Lens AI front-end behaviour (no external libraries; loaded on every page).
(function () {
  "use strict";

  // ---- Input form: show only the panel for the selected input type.
  var form = document.getElementById("check-form");
  if (form) {
    var radios = form.querySelectorAll('input[name="input_type"]');
    var panels = form.querySelectorAll(".panel");
    var btn = document.getElementById("submit-btn");
    var update = function () {
      var selected = form.querySelector('input[name="input_type"]:checked');
      var value = selected ? selected.value : "text";
      panels.forEach(function (p) {
        var active = p.getAttribute("data-panel") === value;
        p.classList.toggle("hidden", !active);
        // Disable inputs in hidden panels so they are not submitted.
        p.querySelectorAll("input, textarea").forEach(function (el) { el.disabled = !active; });
      });
      // Audio/video are transcribed first; the user reviews the transcript before any fact-check.
      if (btn) { btn.textContent = (value === "audio" || value === "video") ? "Transcribe recording" : "Check claims"; }
    };
    radios.forEach(function (r) { r.addEventListener("change", update); });
    update();
    form.addEventListener("submit", function () {
      if (btn) { btn.disabled = true; btn.textContent = "Uploading and starting..."; }
    });
  }

  // ---- Transcript review: put the ticked statements into the text box.
  var useSelected = document.getElementById("use-selected");
  var transcriptBox = document.getElementById("text");
  if (useSelected && transcriptBox) {
    var picks = document.querySelectorAll(".seg-pick");
    var refresh = function () {
      var n = Array.prototype.filter.call(picks, function (p) { return p.checked; }).length;
      useSelected.disabled = n === 0;
      useSelected.textContent = n ? "Check only the " + n + " ticked statement(s)" : "Tick statements to check only those";
    };
    picks.forEach(function (p) { p.addEventListener("change", refresh); });
    useSelected.classList.remove("hidden");
    refresh();
    useSelected.addEventListener("click", function () {
      var chosen = [];
      picks.forEach(function (p) { if (p.checked) { chosen.push(p.getAttribute("data-text")); } });
      if (chosen.length) {
        transcriptBox.value = chosen.join("\n");
        transcriptBox.focus();
      }
    });
    var restore = document.getElementById("restore-transcript");
    if (restore) {
      restore.classList.remove("hidden");
      restore.addEventListener("click", function () {
        transcriptBox.value = transcriptBox.defaultValue;
        picks.forEach(function (p) { p.checked = false; });
        refresh();
      });
    }
  }

  // ---- Progress page: poll the job status until the result is ready.
  var progress = document.getElementById("progress");
  if (progress) {
    var statusUrl = progress.getAttribute("data-status-url");
    var reportUrl = progress.getAttribute("data-report-url");
    var msg = document.getElementById("progress-message");
    var errBox = document.getElementById("progress-error");
    var order = Array.prototype.map.call(progress.querySelectorAll(".stages li"), function (li) {
      return li.getAttribute("data-stage");
    });
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
          if (s.state === "done") { window.location.href = s.next_url || reportUrl; return; }
          if (s.state === "error") { showError("It failed: " + s.error); return; }
          msg.textContent = s.message || "Working...";
          markStage(s.stage);
          setTimeout(poll, 1500);
        })
        .catch(function (e) {
          if (e && e.message === "gone") {
            showError("This job is no longer available (the server may have been restarted). Please submit it again.");
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
