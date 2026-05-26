chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(["candidateProfile"], (result) => {
    if (result.candidateProfile) {
      return;
    }

    chrome.storage.sync.set({
      candidateProfile: {
        fullName: "",
        email: "",
        phone: "",
        location: "",
        linkedin: "",
        github: "",
        website: "",
        yearsExperience: "",
        resumeText: "",
        defaultCoverLetterTone: "professional"
      }
    });
  });
});
