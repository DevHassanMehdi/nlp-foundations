import "./style.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Episode Generator</p>
        <h1>Three Models, One Prompt</h1>
        <p class="subhead">Planner-driven generation with shared episode structure across n-gram, word-LSTM, and a locally hosted transformer model.</p>
      </div>
      <div class="badge">Local demo</div>
    </header>

    <section class="input-panel">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" rows="4" placeholder="A detective arrives in a quiet town..."></textarea>
      <div class="controls">
        <div>
          <label for="temp">Temperature</label>
          <input id="temp" type="number" value="0.72" step="0.02" min="0.6" max="0.95" />
        </div>
        <div>
          <label for="seed">Seed</label>
          <input id="seed" type="number" value="42" min="1" max="99999" />
        </div>
      </div>
      <div class="actions">
        <button id="runBtn">Generate episode</button>
        <button id="sampleBtn" class="ghost">Use sample</button>
      </div>
      <p class="status" id="status">Ready.</p>
    </section>

    <section class="results">
      <div class="card">
        <div class="card-header">
          <h2>N-gram Model</h2>
          <span class="tag">Word-level bigram</span>
        </div>
        <pre id="ngramOut">No output yet.</pre>
      </div>

      <div class="card accent">
        <div class="card-header">
          <h2>Word-LSTM</h2>
          <span class="tag">LSTM, word-level</span>
        </div>
        <pre id="lstmOut">No output yet.</pre>
      </div>

      <div class="card">
        <div class="card-header">
          <h2>Transformer LM</h2>
          <span class="tag">Qwen2.5 Instruct (local)</span>
        </div>
        <pre id="pretrainedOut">No output yet.</pre>
      </div>
    </section>
  </div>
`;

const sampleText = "On a nice, peaceful morning, Janene goes for a walk and has a strange encounter with a bear in the woods.";

const runBtn = document.querySelector("#runBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const promptEl = document.querySelector("#prompt");
const status = document.querySelector("#status");
const ngramOut = document.querySelector("#ngramOut");
const lstmOut = document.querySelector("#lstmOut");
const pretrainedOut = document.querySelector("#pretrainedOut");

sampleBtn.addEventListener("click", () => {
  promptEl.value = sampleText;
});

runBtn.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    status.textContent = "Please enter a prompt.";
    return;
  }

  const temperature = Number(document.querySelector("#temp").value);
  const seed = Number(document.querySelector("#seed").value);

  status.textContent = "Generating...";
  runBtn.disabled = true;

  try {
    const res = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, temperature, seed })
    });

    if (!res.ok) throw new Error("Backend error");

    const data = await res.json();
    ngramOut.textContent = data.ngram;
    lstmOut.textContent = data.word_lstm;
    pretrainedOut.textContent = data.pretrained;
    status.textContent = "Done.";
  } catch (err) {
    status.textContent = "Could not reach backend. Is FastAPI running?";
  } finally {
    runBtn.disabled = false;
  }
});
