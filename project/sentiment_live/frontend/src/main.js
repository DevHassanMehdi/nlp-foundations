import "./style.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Sentiment Live</p>
        <h1>Classic vs Transformer</h1>
        <p class="subhead">Two pipelines. Same text. Instant sentiment side‑by‑side.</p>
      </div>
      <div class="badge">Local demo</div>
    </header>

    <section class="input-panel">
      <label for="inputText">Enter text</label>
      <textarea id="inputText" rows="5" placeholder="Type a sentence or review..."></textarea>
      <div class="actions">
        <button id="runBtn">Analyze sentiment</button>
        <button id="sampleBtn" class="ghost">Use sample</button>
      </div>
      <p class="status" id="status">Ready.</p>
    </section>

    <section class="results">
      <div class="card">
        <div class="card-header">
          <h2>Classic</h2>
          <span class="tag">TF‑IDF + Logistic Regression</span>
        </div>
        <div class="score" id="classicScore">—</div>
        <div class="label" id="classicLabel">No result yet</div>
        <ul class="details">
          <li>Bag‑of‑words features</li>
          <li>Interpretable weights</li>
          <li>Fast inference</li>
        </ul>
      </div>

      <div class="card accent">
        <div class="card-header">
          <h2>Transformer</h2>
          <span class="tag">DistilBERT (SST‑2)</span>
        </div>
        <div class="score" id="transformerScore">—</div>
        <div class="label" id="transformerLabel">No result yet</div>
        <ul class="details">
          <li>Contextual embeddings</li>
          <li>Pretrained language model</li>
          <li>Fine‑tuned for sentiment</li>
        </ul>
      </div>
    </section>
  </div>
`;

const runBtn = document.querySelector("#runBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const inputText = document.querySelector("#inputText");
const status = document.querySelector("#status");

const classicScore = document.querySelector("#classicScore");
const classicLabel = document.querySelector("#classicLabel");
const transformerScore = document.querySelector("#transformerScore");
const transformerLabel = document.querySelector("#transformerLabel");

const sampleText =
  "The story started slowly, but the ending was surprisingly emotional and beautifully written.";

sampleBtn.addEventListener("click", () => {
  inputText.value = sampleText;
  inputText.focus();
});

runBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text) {
    status.textContent = "Please enter some text.";
    return;
  }

  status.textContent = "Analyzing...";
  runBtn.disabled = true;
  try {
    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error("Backend error");
    }

    const data = await res.json();

    classicScore.textContent = (data.classic.score * 100).toFixed(1) + "%";
    classicLabel.textContent = data.classic.label;

    transformerScore.textContent = (data.transformer.score * 100).toFixed(1) + "%";
    transformerLabel.textContent = data.transformer.label;

    status.textContent = "Done.";
  } catch (err) {
    status.textContent = "Could not reach backend. Is FastAPI running?";
  } finally {
    runBtn.disabled = false;
  }
});
