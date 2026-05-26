const JOB_ROLES = {
  "Data Analyst": {
    description:
      "Analyze datasets, create dashboards, build reports, use SQL, Excel, Python, Tableau, Power BI, statistics, and data visualization.",
    skills: [
      "python",
      "sql",
      "excel",
      "tableau",
      "power bi",
      "statistics",
      "data visualization",
      "pandas",
      "dashboard",
      "reporting"
    ]
  },
  "Machine Learning Engineer": {
    description:
      "Build, train, evaluate, and deploy machine learning models using Python, scikit-learn, TensorFlow, PyTorch, NLP, computer vision, APIs, and cloud tools.",
    skills: [
      "python",
      "machine learning",
      "scikit-learn",
      "tensorflow",
      "pytorch",
      "nlp",
      "computer vision",
      "model deployment",
      "api",
      "cloud"
    ]
  },
  "Software Engineer": {
    description:
      "Design and develop applications using data structures, algorithms, Java, Python, JavaScript, React, backend APIs, databases, Git, testing, and system design.",
    skills: [
      "java",
      "python",
      "javascript",
      "react",
      "api",
      "database",
      "git",
      "testing",
      "algorithms",
      "data structures"
    ]
  },
  "Cybersecurity Analyst": {
    description:
      "Monitor systems, detect threats, analyze vulnerabilities, use networking, Linux, SIEM, incident response, risk assessment, and security tools.",
    skills: [
      "networking",
      "linux",
      "siem",
      "incident response",
      "vulnerability",
      "risk assessment",
      "security",
      "firewall",
      "python",
      "threat analysis"
    ]
  },
  "Web Developer": {
    description:
      "Build responsive websites and web apps using HTML, CSS, JavaScript, React, Node.js, APIs, databases, Git, UI design, and accessibility.",
    skills: [
      "html",
      "css",
      "javascript",
      "react",
      "node",
      "api",
      "database",
      "git",
      "ui",
      "accessibility"
    ]
  }
};

const FIELD_MATCHERS = [
  { key: "fullName", patterns: ["full name", "name", "applicant_name"] },
  { key: "email", patterns: ["email", "e-mail"] },
  { key: "phone", patterns: ["phone", "mobile", "telephone"] },
  { key: "location", patterns: ["location", "city", "address"] },
  { key: "linkedin", patterns: ["linkedin"] },
  { key: "github", patterns: ["github"] },
  { key: "website", patterns: ["portfolio", "website", "personal site"] },
  { key: "resumeText", patterns: ["summary", "about", "cover letter", "experience", "additional information"] }
];

const INJECTED_STYLE_ID = "jma-style";
const PANEL_ID = "jma-panel";

const SITE_PRESETS = {
  linkedin: {
    titleSelectors: [
      ".job-details-jobs-unified-top-card__job-title",
      ".top-card-layout__title",
      "h1"
    ],
    companySelectors: [
      ".job-details-jobs-unified-top-card__company-name a",
      ".jobs-unified-top-card__company-name a",
      ".topcard__org-name-link"
    ],
    descriptionSelectors: [
      ".jobs-description-content__text",
      ".jobs-box__html-content",
      "#job-details"
    ]
  },
  indeed: {
    titleSelectors: [
      "h1[data-testid='jobsearch-JobInfoHeader-title']",
      ".jobsearch-JobInfoHeader-title",
      "h1"
    ],
    companySelectors: [
      "[data-testid='inlineHeader-companyName']",
      ".jobsearch-CompanyInfoContainer"
    ],
    descriptionSelectors: [
      "#jobDescriptionText",
      "[data-testid='jobsearch-JobComponent-description']",
      "#jobsearch-JobComponent-description"
    ]
  },
  greenhouse: {
    titleSelectors: ["#header .app-title", ".app-title", "h1"],
    companySelectors: ["#header .company-name", ".company-name", ".company"],
    descriptionSelectors: [
      "#content",
      ".job-post",
      ".section-wrapper"
    ]
  },
  lever: {
    titleSelectors: [
      ".posting-headline h2",
      ".posting-headline .posting-title",
      "h2"
    ],
    companySelectors: [
      ".posting-categories .sort-by-time",
      ".main-header-logo img",
      ".posting-headline"
    ],
    descriptionSelectors: [
      ".posting-description",
      ".section-wrapper .content",
      ".posting-page"
    ]
  }
};

function normalizeText(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9+#.\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  const normalized = normalizeText(text);
  if (!normalized) {
    return [];
  }
  return normalized.split(" ").filter(Boolean);
}

function overlapScore(a, b) {
  const setA = new Set(tokenize(a));
  const setB = new Set(tokenize(b));
  if (setA.size === 0 || setB.size === 0) {
    return 0;
  }

  let intersection = 0;
  setA.forEach((item) => {
    if (setB.has(item)) {
      intersection += 1;
    }
  });

  return intersection / Math.sqrt(setA.size * setB.size);
}

function safeQueryText(selector) {
  const element = document.querySelector(selector);
  if (!element) {
    return "";
  }

  // Support image-based company logos where name is provided in alt text.
  if (element.tagName === "IMG") {
    return (element.getAttribute("alt") || "").trim();
  }

  return (element.textContent || "").trim();
}

function safeMetaContent(selector) {
  const element = document.querySelector(selector);
  return element ? (element.getAttribute("content") || "").trim() : "";
}

function firstLongText(selectors, minLength) {
  for (const selector of selectors) {
    const value = safeQueryText(selector);
    if (value.length >= minLength) {
      return value;
    }
  }
  return "";
}

function detectSitePreset(hostname) {
  if (hostname.includes("linkedin.com")) {
    return "linkedin";
  }
  if (hostname.includes("indeed.")) {
    return "indeed";
  }
  if (hostname.includes("greenhouse.io")) {
    return "greenhouse";
  }
  if (hostname.includes("lever.co")) {
    return "lever";
  }
  return "generic";
}

function fallbackCompanyFromHostname(hostname, sitePreset) {
  if (sitePreset === "greenhouse") {
    const parts = hostname.split(".");
    if (parts.length >= 3 && parts[0] !== "boards") {
      return parts[0].replace(/[-_]/g, " ");
    }
  }

  if (sitePreset === "lever") {
    const parts = hostname.split(".");
    if (parts.length >= 3 && parts[0] !== "jobs") {
      return parts[0].replace(/[-_]/g, " ");
    }
  }

  return "";
}

function extractJobContext() {
  const hostname = window.location.hostname;
  const sitePreset = detectSitePreset(hostname);
  const preset = SITE_PRESETS[sitePreset];

  const titleFromPreset = preset ? firstLongText(preset.titleSelectors, 3) : "";
  const title =
    titleFromPreset ||
    safeQueryText("h1") ||
    safeMetaContent("meta[property='og:title']") ||
    document.title;

  const companyFromPreset = preset ? firstLongText(preset.companySelectors, 2) : "";
  const company =
    companyFromPreset ||
    safeQueryText("[data-testid='inlineHeader-companyName']") ||
    safeQueryText(".topcard__org-name-link") ||
    safeQueryText(".jobsearch-CompanyInfoContainer") ||
    fallbackCompanyFromHostname(hostname, sitePreset) ||
    "Company";

  const defaultDescriptionSelectors = [
    "[data-testid='job-details']",
    ".description__text",
    "#job-details",
    "#jobDescriptionText",
    "article",
    "main"
  ];
  const descriptionSelectors = preset
    ? [...preset.descriptionSelectors, ...defaultDescriptionSelectors]
    : defaultDescriptionSelectors;

  const description =
    firstLongText(descriptionSelectors, 280) ||
    (document.body.innerText || "").slice(0, 14000);

  return {
    title: title || "Job Opportunity",
    company,
    description,
    sourceSite: sitePreset
  };
}

function extractMatchedSkills(description, skills) {
  const cleaned = normalizeText(description);
  return skills.filter((skill) => cleaned.includes(skill));
}

function rankRoles(description) {
  const roleScores = Object.entries(JOB_ROLES).map(([role, data]) => {
    const semanticScore = overlapScore(description, data.description);
    const matchedSkills = extractMatchedSkills(description, data.skills);
    const skillScore = data.skills.length ? matchedSkills.length / data.skills.length : 0;
    const score = Math.round((semanticScore * 0.6 + skillScore * 0.4) * 100);

    return {
      role,
      score,
      matchedSkills,
      missingSkills: data.skills.filter((skill) => !matchedSkills.includes(skill))
    };
  });

  roleScores.sort((a, b) => b.score - a.score);
  return roleScores;
}

function buildResumeBullets(profile, jobContext, roleResult) {
  const years = profile.yearsExperience ? `${profile.yearsExperience}+ years` : "hands-on";
  const topMissing = roleResult.missingSkills.slice(0, 3);
  const missingPhrase = topMissing.length
    ? `with emphasis on ${topMissing.join(", ")}`
    : "aligned to the position requirements";

  return [
    `Delivered ${years} of project execution in ${roleResult.role} style work, ${missingPhrase}.`,
    `Built measurable outcomes by collaborating across teams to support ${jobContext.company} business goals.`,
    `Applied a structured problem-solving approach to improve quality, speed, and operational reliability.`,
    `Maintained clear documentation and communication, enabling faster onboarding and smoother handoffs.`
  ];
}

function buildCoverLetter(profile, jobContext, roleResult) {
  const fullName = profile.fullName || "Candidate";
  const tone = profile.defaultCoverLetterTone || "professional";
  const roleLabel = roleResult.role;

  return [
    `Dear Hiring Team at ${jobContext.company},`,
    "",
    `I am excited to apply for the ${jobContext.title} role. Based on the posting, this opportunity strongly aligns with my background in ${roleLabel} work and my interest in delivering clear, measurable outcomes.`,
    "",
    `My approach is ${tone}, collaborative, and results-focused. I have experience turning ambiguous requirements into practical solutions while maintaining quality and communication across stakeholders.`,
    "",
    `I would value the opportunity to contribute to ${jobContext.company} and help move this role's priorities forward. Thank you for your time and consideration.`,
    "",
    `Sincerely,`,
    fullName
  ].join("\n");
}

function escapeHtml(text) {
  return (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function copyToClipboard(value) {
  navigator.clipboard.writeText(value).catch(() => {});
}

function addStyles() {
  if (document.getElementById(INJECTED_STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = INJECTED_STYLE_ID;
  style.textContent = `
    #${PANEL_ID} {
      position: fixed;
      right: 20px;
      bottom: 20px;
      width: 400px;
      max-height: 80vh;
      overflow: auto;
      z-index: 2147483646;
      background: #ffffff;
      color: #1a1a1a;
      border-radius: 12px;
      border: 1px solid #d9e2ef;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.18);
      padding: 14px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .jma-hidden { display: none; }
    .jma-row { display: flex; gap: 8px; margin: 8px 0; }
    .jma-action {
      border: 0;
      border-radius: 8px;
      padding: 8px 10px;
      background: #eef3f9;
      color: #1a2f47;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }

    .jma-title { margin: 2px 0 10px; font-size: 16px; font-weight: 700; }
    .jma-subtitle { margin: 12px 0 6px; font-size: 13px; font-weight: 700; }
    .jma-text { margin: 0; font-size: 12px; line-height: 1.5; white-space: pre-wrap; }
    .jma-list { margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.5; }
    .jma-score { font-size: 22px; font-weight: 700; margin: 0; }
    .jma-close {
      float: right;
      border: 0;
      background: transparent;
      color: #6b7785;
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
    }
  `;

  document.head.appendChild(style);
}

function ensureWidget() {
  addStyles();

  if (!document.getElementById(PANEL_ID)) {
    const panel = document.createElement("section");
    panel.id = PANEL_ID;
    panel.className = "jma-hidden";
    panel.innerHTML = `
      <button class="jma-close" title="Close panel" aria-label="Close panel">x</button>
      <h2 class="jma-title">Job Match Assistant</h2>
      <div class="jma-row">
        <button class="jma-action" data-action="analyze">Analyze Job</button>
        <button class="jma-action" data-action="autofill">Autofill Form</button>
      </div>
      <div id="jma-results">
        <p class="jma-text">Click Analyze Job to score this posting and generate tailored suggestions.</p>
      </div>
    `;

    panel.querySelector(".jma-close").addEventListener("click", () => {
      panel.classList.add("jma-hidden");
    });

    panel.querySelector("[data-action='analyze']").addEventListener("click", () => {
      runAnalysis().catch(() => {});
    });

    panel.querySelector("[data-action='autofill']").addEventListener("click", () => {
      runAutofill().catch(() => {});
    });

    document.body.appendChild(panel);
  }
}

function renderResults(result) {
  const container = document.getElementById("jma-results");
  if (!container) {
    return;
  }

  const roleRows = result.rankedRoles
    .slice(0, 3)
    .map((item) => `<li>${escapeHtml(item.role)}: ${item.score}% match</li>`)
    .join("");

  const bullets = result.resumeBullets
    .map((bullet) => `<li>${escapeHtml(bullet)}</li>`)
    .join("");

  container.innerHTML = `
    <p class="jma-subtitle">Top Match</p>
    <p class="jma-score">${result.topRole.score}%</p>
    <p class="jma-text">${escapeHtml(result.topRole.role)} for ${escapeHtml(result.jobContext.title)} at ${escapeHtml(result.jobContext.company)}</p>
    <p class="jma-text">Extractor preset: ${escapeHtml(result.jobContext.sourceSite || "generic")}</p>

    <p class="jma-subtitle">Role Ranking</p>
    <ul class="jma-list">${roleRows}</ul>

    <p class="jma-subtitle">Tailored Resume Bullets</p>
    <ul class="jma-list">${bullets}</ul>

    <div class="jma-row">
      <button class="jma-action" data-copy="bullets">Copy Bullets</button>
      <button class="jma-action" data-copy="cover">Copy Cover Letter</button>
    </div>

    <p class="jma-subtitle">Cover Letter Draft</p>
    <p class="jma-text">${escapeHtml(result.coverLetter)}</p>
  `;

  const copyBulletsBtn = container.querySelector("[data-copy='bullets']");
  copyBulletsBtn.addEventListener("click", () => {
    copyToClipboard(result.resumeBullets.map((bullet) => `- ${bullet}`).join("\n"));
  });

  const copyCoverBtn = container.querySelector("[data-copy='cover']");
  copyCoverBtn.addEventListener("click", () => {
    copyToClipboard(result.coverLetter);
  });
}

function profileFromStorage() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["candidateProfile"], (storage) => {
      resolve(storage.candidateProfile || {});
    });
  });
}

async function runAnalysis() {
  ensureWidget();
  const profile = await profileFromStorage();
  const jobContext = extractJobContext();
  const rankedRoles = rankRoles(jobContext.description);
  const topRole = rankedRoles[0];
  const resumeBullets = buildResumeBullets(profile, jobContext, topRole);
  const coverLetter = buildCoverLetter(profile, jobContext, topRole);

  const result = {
    jobContext,
    rankedRoles,
    topRole,
    resumeBullets,
    coverLetter
  };

  renderResults(result);
  return result;
}

function setInputValue(input, value) {
  if (input.readOnly || input.disabled) {
    return false;
  }

  input.focus();
  input.value = value;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
  return true;
}

function matchProfileKey(fieldText) {
  const normalized = normalizeText(fieldText);
  for (const matcher of FIELD_MATCHERS) {
    if (matcher.patterns.some((pattern) => normalized.includes(pattern))) {
      return matcher.key;
    }
  }
  return "";
}

async function runAutofill() {
  const profile = await profileFromStorage();
  let filledCount = 0;

  const elements = Array.from(document.querySelectorAll("input, textarea"));

  elements.forEach((element) => {
    const label =
      (element.labels && element.labels.length ? element.labels[0].innerText : "") ||
      element.placeholder ||
      element.name ||
      element.id ||
      "";

    const profileKey = matchProfileKey(label);
    if (!profileKey || !profile[profileKey]) {
      return;
    }

    if (setInputValue(element, profile[profileKey])) {
      filledCount += 1;
    }
  });

  const panel = document.getElementById(PANEL_ID);
  const results = document.getElementById("jma-results");
  if (panel && results) {
    panel.classList.remove("jma-hidden");
    results.innerHTML = `<p class="jma-text">Autofill complete. Updated ${filledCount} fields on this page.</p>`;
  }

  return { filledCount };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "PING_CONTENT") {
    sendResponse({ ok: true });
    return false;
  }

  if (message?.type === "OPEN_PANEL") {
    ensureWidget();
    const panel = document.getElementById(PANEL_ID);
    if (panel) {
      panel.classList.remove("jma-hidden");
    }
    sendResponse({ ok: true });
    return false;
  }

  if (message?.type === "ANALYZE_JOB") {
    runAnalysis()
      .then((result) => {
        sendResponse({ ok: true, result: { topRole: result.topRole, title: result.jobContext.title } });
      })
      .catch((error) => {
        sendResponse({ ok: false, error: String(error) });
      });
    return true;
  }

  if (message?.type === "AUTOFILL_FORM") {
    runAutofill()
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({ ok: false, error: String(error) }));
    return true;
  }

  return false;
});

ensureWidget();
