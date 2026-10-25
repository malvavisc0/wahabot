/* Progressive enhancement only: dirty-form warnings and focus helpers. */

(function () {
  "use strict";

  function markDirty(form) {
    if (form.dataset.dirty === "true") return;
    form.dataset.dirty = "true";
    form.classList.add("is-dirty");
    var feedback = form.querySelector(".feedback.unsaved");
    if (feedback) feedback.hidden = false;
  }

  function clearDirty(form) {
    form.dataset.dirty = "false";
    form.classList.remove("is-dirty");
    var feedback = form.querySelector(".feedback.unsaved");
    if (feedback) feedback.hidden = true;
  }

  document.querySelectorAll("form").forEach(function (form) {
    if (form.dataset.noDirty === "true") return;
    form.addEventListener("input", function () {
      markDirty(form);
    });
    form.addEventListener("change", function () {
      markDirty(form);
    });
    form.addEventListener("submit", function () {
      clearDirty(form);
    });
  });

  window.addEventListener("beforeunload", function (event) {
    var dirty = document.querySelector('form.is-dirty[data-warn-leave="true"]');
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  if (window.location.hash) {
    var target = document.getElementById(window.location.hash.slice(1));
    if (target) target.focus();
  }

  var setupStatus = document.querySelector("[data-setup-status]");
  if (setupStatus) {
    window.setInterval(function () {
      window
        .fetch("/setup/status", { credentials: "same-origin" })
        .then(function (response) {
          return response.ok ? response.json() : null;
        })
        .then(function (state) {
          if (!state) return;
          var label = document.querySelector("[data-setup-status]");
          if (label) label.textContent = state.status;
          var qr = document.getElementById("setup-qr");
          if (qr && state.qr_data) qr.src = state.qr_data;
          if (state.completed) window.location.assign("/app/history");
          if (state.restarted) window.location.reload();
        })
        .catch(function () {});
    }, 5000);
  }

  document.querySelectorAll("[data-focus-target]").forEach(function (control) {
    control.addEventListener("click", function () {
      var target = document.getElementById(control.dataset.focusTarget);
      if (target) {
        window.setTimeout(function () {
          target.focus();
        }, 0);
      }
    });
  });
})();
