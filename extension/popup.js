async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function sendToContent(message) {
  const tab = await getActiveTab();
  if (!tab?.id) {
    throw new Error("No active tab available.");
  }

  return chrome.tabs.sendMessage(tab.id, message);
}

function setStatus(text) {
  const statusEl = document.getElementById("status");
  statusEl.textContent = text;
}

async function runAction(type) {
  try {
    const response = await sendToContent({ type });

    if (!response?.ok) {
      throw new Error(response?.error || "Action failed.");
    }

    if (type === "ANALYZE_JOB") {
      setStatus(
        `Top match: ${response.result.topRole.role} (${response.result.topRole.score}%) for ${response.result.title}.`
      );
      return;
    }

    if (type === "AUTOFILL_FORM") {
      setStatus(`Autofill updated ${response.result.filledCount} fields on this page.`);
      return;
    }

    setStatus("Panel opened on page.");
  } catch (error) {
    setStatus(`Could not complete request: ${error.message}`);
  }
}

document.getElementById("analyzeBtn").addEventListener("click", () => {
  runAction("ANALYZE_JOB");
});

document.getElementById("autofillBtn").addEventListener("click", () => {
  runAction("AUTOFILL_FORM");
});

document.getElementById("openPanelBtn").addEventListener("click", () => {
  runAction("OPEN_PANEL");
});
