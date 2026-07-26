(function () {
  /* Runs inside components.html iframe — operate on the parent Streamlit document. */
  var doc;
  try {
    doc = window.parent && window.parent.document ? window.parent.document : document;
  } catch (err) {
    doc = document;
  }
  var win = doc.defaultView || window.parent || window;

  /* Bump when close/open logic changes — must rebind after remount */
  if (win.__qvacUiPortalV7) return;
  win.__qvacUiPortalV7 = true;

  function overlayFor(ck) {
    if (!ck) return null;
    if (ck._qvacOverlay && ck._qvacOverlay.isConnected) return ck._qvacOverlay;
    var o = ck.nextElementSibling;
    if (o && o.classList && o.classList.contains("fs-overlay")) {
      ck._qvacOverlay = o;
      return o;
    }
    var id = ck.id;
    if (id) {
      var closeEl = doc.querySelector(
        'button.fs-close[data-fs="' +
          id +
          '"], label.fs-close[for="' +
          id +
          '"]'
      );
      var card = closeEl && closeEl.closest ? closeEl.closest(".fs-overlay") : null;
      if (card) {
        ck._qvacOverlay = card;
        return card;
      }
    }
    return ck._qvacOverlay || null;
  }

  function scrollport(overlay) {
    if (!overlay) return null;
    return (
      overlay.querySelector(".fs-scroll") ||
      overlay.querySelector(".guide-body") ||
      overlay.querySelector(".fs-pre")
    );
  }

  function forceHide(overlay) {
    if (!overlay) return;
    overlay.classList.remove("qvac-fs-open");
    overlay.setAttribute("hidden", "");
    overlay.style.setProperty("display", "none", "important");
    overlay.style.setProperty("visibility", "hidden", "important");
    overlay.style.setProperty("pointer-events", "none", "important");
  }

  function forceShow(overlay) {
    if (!overlay) return;
    overlay.classList.add("qvac-fs-open");
    overlay.removeAttribute("hidden");
    overlay.style.setProperty("display", "flex", "important");
    overlay.style.setProperty("visibility", "visible", "important");
    overlay.style.setProperty("pointer-events", "auto", "important");
  }

  function park(overlay) {
    if (!overlay || overlay.dataset.qvacParked === "1") return;
    var ph = doc.createComment("qvac-fs-ph");
    if (overlay.parentNode) overlay.parentNode.insertBefore(ph, overlay);
    doc.body.appendChild(overlay);
    overlay.dataset.qvacParked = "1";
    overlay._qvacPh = ph;
  }

  function unpark(overlay) {
    if (!overlay || overlay.dataset.qvacParked !== "1") return;
    var ph = overlay._qvacPh;
    if (ph && ph.parentNode) {
      ph.parentNode.insertBefore(overlay, ph);
      ph.parentNode.removeChild(ph);
    }
    overlay.dataset.qvacParked = "0";
    overlay._qvacPh = null;
  }

  function closeOverlay(ck) {
    if (!ck) return;
    ck.checked = false;
    var overlay = overlayFor(ck);
    if (!overlay) return;
    forceHide(overlay);
    unpark(overlay);
  }

  function openOverlay(ck) {
    if (!ck) return;
    ck.checked = true;
    var overlay = overlayFor(ck);
    if (!overlay) return;
    park(overlay);
    forceShow(overlay);
    var sc = scrollport(overlay);
    if (sc) {
      sc.style.overflowY = "auto";
      sc.style.webkitOverflowScrolling = "touch";
    }
  }

  function sync(ck) {
    if (!ck) return;
    if (ck.checked) openOverlay(ck);
    else closeOverlay(ck);
  }

  function checkboxIdFromClose(el) {
    if (!el) return null;
    return el.getAttribute("data-fs") || el.getAttribute("for") || null;
  }

  function hideAllClosed() {
    doc.querySelectorAll(".fs-overlay").forEach(function (overlay) {
      var open = overlay.classList.contains("qvac-fs-open");
      var closeEl = overlay.querySelector("button.fs-close[data-fs], label.fs-close[for]");
      var id = checkboxIdFromClose(closeEl);
      var ck = id ? doc.getElementById(id) : null;
      if (ck && ck.checked) {
        /* keep open — do not thrash */
        if (!open) openOverlay(ck);
      } else {
        forceHide(overlay);
        if (overlay.dataset.qvacParked === "1") unpark(overlay);
      }
    });
  }

  doc.addEventListener(
    "change",
    function (e) {
      var t = e.target;
      if (t && t.classList && t.classList.contains("fs-ck")) sync(t);
    },
    true
  );

  /* ✕ must NOT be a bare label[for] toggle — native label re-checks after we uncheck */
  doc.addEventListener(
    "click",
    function (e) {
      var t = e.target;
      if (!t || !t.closest) return;

      var closeEl = t.closest("button.fs-close, label.fs-close");
      if (closeEl) {
        e.preventDefault();
        e.stopPropagation();
        if (e.stopImmediatePropagation) e.stopImmediatePropagation();
        var id = checkboxIdFromClose(closeEl);
        if (!id) return;
        var ck = doc.getElementById(id);
        closeOverlay(ck);
        return;
      }

      /* Click dimmed backdrop (the overlay itself, not the card) closes */
      if (t.classList && t.classList.contains("fs-overlay") && t.classList.contains("qvac-fs-open")) {
        var closeBtn = t.querySelector("button.fs-close[data-fs], label.fs-close[for]");
        var oid = checkboxIdFromClose(closeBtn);
        if (oid) closeOverlay(doc.getElementById(oid));
      }
    },
    true
  );

  doc.addEventListener(
    "keydown",
    function (e) {
      if (e.key !== "Escape") return;
      var open = doc.querySelector(".fs-overlay.qvac-fs-open");
      if (!open) return;
      var closeEl = open.querySelector("button.fs-close[data-fs], label.fs-close[for]");
      var id = checkboxIdFromClose(closeEl);
      if (!id) return;
      e.preventDefault();
      closeOverlay(doc.getElementById(id));
    },
    true
  );

  function syncOpenFullscreenText() {
    doc.querySelectorAll(".stream-out[data-panel]").forEach(function (src) {
      var uid = src.getAttribute("data-panel");
      if (!uid) return;
      var ck = doc.getElementById("fs_" + uid);
      if (!ck || !ck.checked) return;
      var overlay = overlayFor(ck);
      var pre = overlay && overlay.querySelector(".fs-pre");
      var sc = scrollport(overlay);
      if (!pre || !sc) return;
      if (pre.getAttribute("data-qvac-src") !== src.innerHTML) {
        var nearBottom =
          sc.scrollHeight - sc.scrollTop - sc.clientHeight < 96;
        pre.innerHTML = src.innerHTML;
        pre.setAttribute("data-qvac-src", src.innerHTML);
        if (nearBottom) sc.scrollTop = sc.scrollHeight;
      }
    });
  }

  hideAllClosed();
  try {
    new MutationObserver(function (muts) {
      var added = false;
      for (var i = 0; i < muts.length; i++) {
        if (muts[i].addedNodes && muts[i].addedNodes.length) {
          added = true;
          break;
        }
      }
      if (added) hideAllClosed();
      syncOpenFullscreenText();
    }).observe(doc.body, { childList: true, subtree: true, characterData: true });
  } catch (err) {}
  setInterval(function () {
    hideAllClosed();
    syncOpenFullscreenText();
  }, 800);
})();
