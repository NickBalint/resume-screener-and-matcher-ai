const DEFAULT_PROFILE = {
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
};

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function fillForm(profile) {
  const form = document.getElementById("profileForm");
  Object.keys(DEFAULT_PROFILE).forEach((key) => {
    if (form.elements[key]) {
      form.elements[key].value = profile[key] || "";
    }
  });
}

function readForm() {
  const form = document.getElementById("profileForm");
  const profile = { ...DEFAULT_PROFILE };

  Object.keys(DEFAULT_PROFILE).forEach((key) => {
    if (form.elements[key]) {
      profile[key] = String(form.elements[key].value || "").trim();
    }
  });

  return profile;
}

chrome.storage.sync.get(["candidateProfile"], (storage) => {
  fillForm({ ...DEFAULT_PROFILE, ...(storage.candidateProfile || {}) });
});

document.getElementById("profileForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const profile = readForm();

  chrome.storage.sync.set({ candidateProfile: profile }, () => {
    setStatus("Profile saved.");
  });
});
