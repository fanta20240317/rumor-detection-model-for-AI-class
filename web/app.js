const sampleText = document.querySelector("#sampleText");
const topK = document.querySelector("#topK");
const useLlm = document.querySelector("#useLlm");
const predictButton = document.querySelector("#predictButton");
const clearButton = document.querySelector("#clearButton");
const formMessage = document.querySelector("#formMessage");
const modelStatus = document.querySelector("#modelStatus");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const thresholdText = document.querySelector("#thresholdText");
const labelBadge = document.querySelector("#labelBadge");
const confidenceText = document.querySelector("#confidenceText");
const probabilityText = document.querySelector("#probabilityText");
const probabilityFill = document.querySelector("#probabilityFill");
const explanationText = document.querySelector("#explanationText");
const llmBlock = document.querySelector("#llmBlock");
const llmExplanationText = document.querySelector("#llmExplanationText");
const factorList = document.querySelector("#factorList");
const baselineList = document.querySelector("#baselineList");
const keywordList = document.querySelector("#keywordList");
const structureList = document.querySelector("#structureList");
const guardText = document.querySelector("#guardText");
const caseList = document.querySelector("#caseList");

function formatPercent(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return `${(numeric * 100).toFixed(digits)}%`;
}

function formatScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return numeric.toFixed(4);
}

function setMessage(text, type = "info") {
  formMessage.textContent = text;
  formMessage.classList.toggle("error", type === "error");
}

function setStatus(text, type = "info") {
  modelStatus.textContent = text;
  modelStatus.className = "status-pill";
  if (type !== "info") modelStatus.classList.add(type);
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function appendTextElement(parent, tagName, text, className) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (status.model_exists && status.train_exists) {
      const llmReady = status.llm_evidence?.configured;
      setStatus(
        llmReady ? "模型和大模型接口已就绪" : "模型文件已就绪，大模型未配置",
        "ready",
      );
      configureLlmToggle(llmReady);
    } else {
      const missing = [];
      if (!status.model_exists) missing.push("模型");
      if (!status.train_exists) missing.push("训练集");
      setStatus(`缺少${missing.join("和")}文件`, "error");
      configureLlmToggle(false);
    }
  } catch (error) {
    setStatus(`状态检查失败：${error.message}`, "error");
    configureLlmToggle(false);
  }
}

function configureLlmToggle(enabled) {
  useLlm.disabled = false;
  useLlm.checked = true;
  useLlm.closest(".toggle-field").classList.toggle("disabled", !enabled);
  useLlm.title = enabled
    ? "调用学校大模型 API 生成自然语言解释"
    : "默认会尝试生成 LLM 证据；未配置学校 API 时会返回未配置提示";
}

async function predict() {
  const text = sampleText.value.trim();
  if (!text) {
    setMessage("请输入一段需要判断的文本。", "error");
    sampleText.focus();
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "判断中";
  setMessage("正在调用本地模型...");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        top_k: topK.value || null,
        use_llm_evidence: useLlm.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "预测失败");
    }
    renderResult(payload);
    setMessage("判断完成。");
    setStatus("模型已加载并可用", "ready");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "开始判断";
  }
}

function renderResult(result) {
  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const labelClass = result.label_name === "rumor" ? "rumor" : "non-rumor";
  labelBadge.className = `label-badge ${labelClass}`;
  labelBadge.textContent = result.label_name === "rumor" ? "谣言" : "非谣言";
  confidenceText.textContent = formatPercent(result.confidence);
  probabilityText.textContent = formatPercent(result.prob_rumor);
  thresholdText.textContent = `阈值 ${formatScore(result.threshold)}`;
  probabilityFill.style.width = `${Math.max(0, Math.min(1, result.prob_rumor)) * 100}%`;
  probabilityFill.classList.toggle("rumor", result.label_name === "rumor");
  explanationText.textContent = result.explanation || "模型没有返回解释。";
  renderLlmEvidence(result.llm_evidence);

  renderFactors(result.evidence?.decision_factors || []);
  renderBaseline(result.evidence?.baseline || result);
  renderKeywords(result.evidence?.keyword_contributions || []);
  renderStructure(result.evidence?.claim_structure_features || {});
  renderGuard(result.evidence?.probability_guard || {});
  renderCases(result.evidence?.retrieved_cases || []);
}

function renderLlmEvidence(llmEvidence) {
  if (!llmEvidence) {
    llmBlock.classList.add("hidden");
    llmExplanationText.textContent = "";
    return;
  }

  llmBlock.classList.remove("hidden");
  if (llmEvidence.available) {
    llmExplanationText.textContent = llmEvidence.explanation;
  } else if (llmEvidence.enabled === false) {
    llmExplanationText.textContent = "本次请求关闭了 LLM 证据生成。";
  } else {
    llmExplanationText.textContent = `LLM 证据暂不可用：${llmEvidence.error}`;
  }
}

function renderFactors(factors) {
  clearNode(factorList);
  if (!factors.length) {
    appendTextElement(factorList, "p", "暂无融合信号。", "muted");
    return;
  }

  factors.forEach((factor) => {
    const item = document.createElement("article");
    item.className = "factor-item";

    const top = document.createElement("div");
    top.className = "factor-top";
    appendTextElement(top, "strong", factor.name || "signal");
    appendTextElement(
      top,
      "span",
      `score ${formatScore(factor.score)} / weight ${formatScore(factor.weight)}`,
      "muted",
    );
    item.appendChild(top);

    appendTextElement(item, "p", factor.description || "模型融合信号");
    factorList.appendChild(item);
  });
}

function renderBaseline(baseline) {
  clearNode(baselineList);
  const rows = [
    ["基线标签", baseline.label_name === "rumor" ? "谣言" : "非谣言"],
    ["基线谣言概率", formatPercent(baseline.prob_rumor ?? baseline.baseline_prob_rumor)],
    ["基线阈值", formatScore(baseline.threshold)],
  ];

  rows.forEach(([name, value]) => {
    appendTextElement(baselineList, "dt", name);
    appendTextElement(baselineList, "dd", value);
  });
}

function renderKeywords(keywords) {
  clearNode(keywordList);
  if (!keywords.length) {
    appendTextElement(keywordList, "span", "暂无关键词证据", "chip");
    return;
  }

  keywords.forEach((item) => {
    appendTextElement(keywordList, "span", item.term || String(item), "chip");
  });
}

function renderStructure(features) {
  clearNode(structureList);
  const signals = features.signals || [];
  if (!signals.length) {
    appendTextElement(
      structureList,
      "span",
      `结构分 ${formatScore(features.structure_score)}`,
      "chip",
    );
    return;
  }

  appendTextElement(
    structureList,
    "span",
    `结构分 ${formatScore(features.structure_score)}`,
    "chip",
  );
  signals.forEach((signal) => appendTextElement(structureList, "span", signal, "chip"));
}

function renderGuard(guard) {
  if (!guard.applied) {
    guardText.textContent = "本次没有触发概率保护规则。";
    return;
  }

  const before = formatPercent(guard.prob_before);
  const after = formatPercent(guard.prob_after);
  const ruleNames = (guard.applied_rules || []).map((rule) => rule.name).join("、");
  guardText.textContent = `已触发 ${ruleNames || guard.type}，谣言概率从 ${before} 调整到 ${after}。`;
}

function renderCases(cases) {
  clearNode(caseList);
  if (!cases.length) {
    appendTextElement(caseList, "p", "没有检索到相似训练案例。", "muted");
    return;
  }

  cases.forEach((item, index) => {
    const article = document.createElement("article");
    article.className = "case-item";

    const meta = document.createElement("div");
    meta.className = "case-meta";
    appendTextElement(meta, "span", `#${index + 1}`);
    appendTextElement(meta, "span", item.label === 1 ? "谣言" : "非谣言");
    appendTextElement(meta, "span", `相似度 ${formatScore(item.score)}`);
    if (item.event !== undefined) {
      appendTextElement(meta, "span", `事件 ${item.event}`);
    }
    article.appendChild(meta);

    appendTextElement(article, "p", item.text || "");
    caseList.appendChild(article);
  });
}

predictButton.addEventListener("click", predict);
sampleText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    predict();
  }
});

clearButton.addEventListener("click", () => {
  sampleText.value = "";
  setMessage("");
  sampleText.focus();
});

document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", () => {
    sampleText.value = button.dataset.sample || "";
    sampleText.focus();
  });
});

loadStatus();
