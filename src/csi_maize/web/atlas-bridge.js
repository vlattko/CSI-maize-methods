"use strict";

// Keep citations outside the embedded atlas and remove opener access.
(() => {
  document.addEventListener("click", (event) => {
    const anchor = event.target.closest && event.target.closest("a[href]");
    const href = anchor && anchor.getAttribute("href");
    // srcdoc inherits the parent URL as its base; handle local fragments here.
    if (href && href.startsWith("#") && window.parent !== window) {
      const destination = document.getElementById(href.slice(1));
      if (destination) {
        event.preventDefault();
        destination.scrollIntoView({behavior: "smooth", block: "start"});
      }
      return;
    }
    if (href && /^https?:\/\//.test(href)) {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    }
  }, true);

  const button = document.getElementById("download");
  const text = document.querySelector("#methods pre");
  if (button && text && window.parent !== window) {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      window.parent.postMessage({
        type: "csi-maize-investigation-export",
        text: text.textContent,
      }, "*");
    }, true);
  }
})();
