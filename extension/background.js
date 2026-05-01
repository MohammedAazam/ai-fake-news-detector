chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkNewsValidator",
    title: "Check with News Validator",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkNewsValidator" && info.selectionText) {
    chrome.storage.local.set({
      pendingText: info.selectionText.trim(),
      pendingTimestamp: Date.now()
    }, () => {
      chrome.action.openPopup().catch(() => {
        chrome.action.setBadgeText({ text: "!" });
        chrome.action.setBadgeBackgroundColor({ color: "#2563eb" });
      });
    });
  }
});