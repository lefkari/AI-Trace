/*
  AI TRACE V4.4
  Human False-Positive Defense
  ------------------------------------
  - TMR + ModernBERT ensemble
  - Calibrated score
  - Evidence quality
  - Abstention / INCONCLUSIVE logic
  - Segment consistency defense
  - Benchmark storage
  - Correct 3-outcome evaluation:
      AI / HUMAN / ABSTAIN
  - Coverage + selective accuracy
  - False-positive monitoring
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

/* =========================================================
   CONFIG
========================================================= */

env.allowLocalModels = false;
env.useBrowserCache = true;

const VERSION = '4.4';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCHMARK_KEY =
  'aiTraceBenchmarkV44';

const HISTORY_KEY =
  'aiTraceHistoryV44';

let tmrClassifier = null;
let modernClassifier = null;

const $ = id =>
  document.getElementById(id);

const text = $('text');

/* =========================================================
   UI
========================================================= */

text.oninput = () => {

  const words =
    text.value.trim()
      ? text.value.trim().split(/\s+/).length
      : 0;

  $('count').textContent =
    `${words} words`;
};

$('clear').onclick = () => {

  text.value = '';

  text.oninput();

  $('report')
    ?.classList
    .add('hidden');
};

$('demo').onclick = () => {

  text.value = `Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

One of the most important advantages of artificial intelligence is its ability to process large amounts of information quickly. Organizations can use AI-powered tools to identify patterns, automate repetitive processes, and support better decision-making.

However, the growing use of artificial intelligence also creates important challenges. AI-generated information can sometimes be inaccurate, misleading, or difficult to distinguish from content created by humans.

The future will therefore require more than simply developing increasingly powerful artificial intelligence systems. Society will also need technologies that provide transparency, verification, and evidence about how digital content was created or modified.`;

  text.oninput();
};

$('scan').onclick = run;

/* =========================================================
   PROGRESS
========================================================= */

function progress(
  percent,
  message
) {

  $('progress')
    ?.classList
    .remove('hidden');

  if ($('bar')) {
    $('bar').style.width =
      percent + '%';
  }

  if ($('progressText')) {
    $('progressText').textContent =
      message;
  }
}

/* =========================================================
   LANGUAGE
========================================================= */

function detectLanguage(value) {

  const latin =
    (
      value.match(/[A-Za-z]/g) || []
    ).length;

  const letters =
    (
      value.match(/\p{L}/gu) || []
    ).length;

  if (!letters) {
    return 'Unknown';
  }

  return latin / letters > 0.80
    ? 'English'
    : 'Non-English';
}

/* =========================================================
   DOCUMENT PROFILE
========================================================= */

function createProfile(value) {

  const words =
    value
      .trim()
      .split(/\s+/)
      .filter(Boolean);

  const sentences =
    value
      .split(/[.!?]+/)
      .map(x => x.trim())
      .filter(Boolean);

  const cleaned =
    words
      .map(
        w =>
          w
            .toLowerCase()
            .replace(
              /[^\p{L}\p{N}]/gu,
              ''
            )
      )
      .filter(Boolean);

  const avg =
    words.length /
    Math.max(
      1,
      sentences.length
    );

  const lexical =
    new Set(cleaned).size /
    Math.max(
      1,
      cleaned.length
    );

  const sentenceLengths =
    sentences.map(
      s =>
        s
          .split(/\s+/)
          .filter(Boolean)
          .length
    );

  const mean =
    sentenceLengths.reduce(
      (a, b) => a + b,
      0
    ) /
    Math.max(
      1,
      sentenceLengths.length
    );

  const variance =
    sentenceLengths.reduce(
      (sum, n) =>
        sum +
        (n - mean) ** 2,
      0
    ) /
    Math.max(
      1,
      sentenceLengths.length
    );

  const transitions =
    (
      value.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless)\b/gi
      ) || []
    ).length;

  return {
    words: words.length,
    sentences: sentences.length,
    avg,
    lexical,
    variance,
    transitions
  };
}

/* =========================================================
   CHUNKS
========================================================= */

function chunkText(
  value,
  maxChars = 1450
) {

  const sentences =
    value.match(
      /[^.!?]+[.!?]+|[^.!?]+$/g
    ) || [value];

  const chunks = [];

  let current = '';

  for (
    const sentence
    of sentences
  ) {

    if (
      (
        current +
        sentence
      ).length >
        maxChars &&
      current
    ) {

      chunks.push(
        current.trim()
      );

      current =
        sentence;

    } else {

      current +=
        sentence;
    }
  }

  if (current.trim()) {

    chunks.push(
      current.trim()
    );
  }

  return chunks
    .filter(Boolean)
    .slice(0, 8);
}

/* =========================================================
   MODELS
========================================================= */

async function loadTMR() {

  if (tmrClassifier) {
    return tmrClassifier;
  }

  $('modelState').textContent =
    'Loading TMR…';

  progress(
    10,
    'Loading Quick Scan model…'
  );

  tmrClassifier =
    await pipeline(
      'text-classification',
      TMR_MODEL,
      {
        dtype: 'q8'
      }
    );

  return tmrClassifier;
}

async function loadModern() {

  if (modernClassifier) {
    return modernClassifier;
  }

  $('modelState').textContent =
    'Loading ModernBERT…';

  progress(
    58,
    'Loading Deep Scan model…'
  );

  modernClassifier =
    await pipeline(
      'text-classification',
      MODERN_MODEL,
      {
        dtype: 'q4f16'
      }
    );

  return modernClassifier;
}

/* =========================================================
   MODEL OUTPUT NORMALIZATION
========================================================= */

function aiProbability(output) {

  const results =
    (
      Array.isArray(output)
        ? output
        : [output]
    ).flat();

  let ai = null;
  let human = null;

  for (
    const item
    of results
  ) {

    const label =
      String(
        item.label || ''
      )
        .toLowerCase();

    const score =
      Number(
        item.score
      ) || 0;

    if (
      label.includes('ai') ||
      label.includes('machine') ||
      label.includes('generated') ||
      label === 'label_1'
    ) {

      ai =
        Math.max(
          ai ?? 0,
          score
        );
    }

    if (
      label.includes('human') ||
      label === 'label_0'
    ) {

      human =
        Math.max(
          human ?? 0,
          score
        );
    }
  }

  if (ai !== null) {
    return ai;
  }

  if (human !== null) {
    return 1 - human;
  }

  if (
    results.length >= 2
  ) {

    return Number(
      results[1]?.score ?? 0.5
    );
  }

  return 0.5;
}

async function classify(
  model,
  value
) {

  const output =
    await model(
      value,
      {
        top_k: null,
        truncation: true
      }
    );

  return Math.round(
    aiProbability(output) *
    100
  );
}

/* =========================================================
   STATISTICS
========================================================= */

function mean(values) {

  if (!values.length) {
    return 0;
  }

  return (
    values.reduce(
      (a, b) => a + b,
      0
    ) /
    values.length
  );
}

function std(values) {

  if (!values.length) {
    return 0;
  }

  const m =
    mean(values);

  return Math.sqrt(
    mean(
      values.map(
        x =>
          (x - m) ** 2
      )
    )
  );
}

/* =========================================================
   RAW ENSEMBLE
========================================================= */

function rawEnsemble(
  tmr,
  modern
) {

  return Math.round(
    (
      tmr +
      modern
    ) /
    2
  );
}

/* =========================================================
   EVIDENCE QUALITY
========================================================= */

function calculateEvidenceQuality({
  modelGap,
  segmentDeviation,
  segmentRange,
  words,
  language,
  modelsActive
}) {

  let quality = 100;

  /*
    Penalize model disagreement.
  */

  quality -=
    Math.min(
      30,
      modelGap * 0.8
    );

  /*
    Penalize instability.
  */

  quality -=
    Math.min(
      30,
      segmentDeviation * 0.65
    );

  quality -=
    Math.min(
      30,
      segmentRange * 0.35
    );

  /*
    Short text penalty.
  */

  if (words < 150) {
    quality -= 10;
  }

  /*
    Language mismatch.
  */

  if (
    language !== 'English'
  ) {

    quality -= 20;
  }

  /*
    Missing model.
  */

  if (
    modelsActive < 2
  ) {

    quality -= 30;
  }

  return Math.max(
    5,
    Math.min(
      100,
      Math.round(
        quality
      )
    )
  );
}

/* =========================================================
   CALIBRATION

   V4.4 intentionally pulls unstable
   predictions toward 50%.
========================================================= */

function calibrate(
  raw,
  evidenceQuality
) {

  const reliability =
    evidenceQuality /
    100;

  /*
    50 = neutral / unknown.

    Strong evidence:
      calibrated ≈ raw.

    Weak evidence:
      calibrated → 50.
  */

  const calibrated =
    50 +
    (
      raw -
      50
    ) *
    reliability;

  return Math.round(
    Math.max(
      0,
      Math.min(
        100,
        calibrated
      )
    )
  );
}

/* =========================================================
   V4.4 ABSTENTION RULES
========================================================= */

function verdictV44({
  calibrated,
  evidenceQuality,
  modelGap,
  segmentRange,
  segmentDeviation,
  modelsActive
}) {

  /*
    HARD ABSTENTION.

    These rules specifically defend
    against human false positives.
  */

  if (
    modelsActive < 2
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    evidenceQuality < 50
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    modelGap >= 35
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    segmentRange > 60
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    segmentDeviation > 28
  ) {

    return 'INCONCLUSIVE';
  }

  /*
    Strong AI requires very strong,
    consistent evidence.
  */

  if (
    calibrated >= 85 &&
    evidenceQuality >= 75 &&
    segmentRange <= 35
  ) {

    return 'Strong AI evidence';
  }

  if (
    calibrated >= 70 &&
    evidenceQuality >= 60 &&
    segmentRange <= 50
  ) {

    return 'Likely AI';
  }

  /*
    Human verdict also requires
    acceptable evidence quality.
  */

  if (
    calibrated <= 15 &&
    evidenceQuality >= 70
  ) {

    return 'Strong human evidence';
  }

  if (
    calibrated <= 30 &&
    evidenceQuality >= 55
  ) {

    return 'Likely human';
  }

  return 'INCONCLUSIVE';
}

/* =========================================================
   BENCHMARK STORAGE
========================================================= */

function loadBenchmark() {

  try {

    return JSON.parse(
      localStorage.getItem(
        BENCHMARK_KEY
      ) || '[]'
    );

  } catch {

    return [];
  }
}

function saveBenchmark(
  data
) {

  localStorage.setItem(
    BENCHMARK_KEY,
    JSON.stringify(
      data
    )
  );
}

function nextID(
  truth,
  records
) {

  const prefix =
    truth === 'AI'
      ? 'A'
      : 'H';

  const count =
    records.filter(
      r =>
        r.groundTruth === truth
    ).length + 1;

  return (
    prefix +
    '-' +
    String(count)
      .padStart(
        3,
        '0'
      )
  );
}

/* =========================================================
   BENCHMARK PREDICTION
========================================================= */

function benchmarkPrediction(
  record
) {

  if (
    record.verdict ===
      'Strong AI evidence' ||
    record.verdict ===
      'Likely AI'
  ) {

    return 'AI';
  }

  if (
    record.verdict ===
      'Strong human evidence' ||
    record.verdict ===
      'Likely human'
  ) {

    return 'HUMAN';
  }

  return 'ABSTAIN';
}

/* =========================================================
   BENCHMARK METRICS — CORRECT 3-OUTCOME LOGIC
========================================================= */

function benchmarkMetrics() {

  const records =
    loadBenchmark()
      .filter(
        r =>
          r.groundTruth === 'AI' ||
          r.groundTruth === 'HUMAN'
      );

  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;

  let abstainAI = 0;
  let abstainHuman = 0;

  for (
    const record
    of records
  ) {

    const prediction =
      benchmarkPrediction(
        record
      );

    if (
      prediction === 'ABSTAIN'
    ) {

      if (
        record.groundTruth ===
          'AI'
      ) {

        abstainAI++;

      } else {

        abstainHuman++;
      }

      continue;
    }

    if (
      record.groundTruth === 'AI' &&
      prediction === 'AI'
    ) {

      TP++;
    }

    if (
      record.groundTruth === 'HUMAN' &&
      prediction === 'HUMAN'
    ) {

      TN++;
    }

    if (
      record.groundTruth === 'HUMAN' &&
      prediction === 'AI'
    ) {

      FP++;
    }

    if (
      record.groundTruth === 'AI' &&
      prediction === 'HUMAN'
    ) {

      FN++;
    }
  }

  const total =
    records.length;

  const decided =
    TP + TN + FP + FN;

  const abstained =
    abstainAI +
    abstainHuman;

  const coverage =
    total
      ? decided / total
      : 0;

  const selectiveAccuracy =
    decided
      ? (
          TP + TN
        ) /
        decided
      : 0;

  const precision =
    TP + FP
      ? TP /
        (
          TP + FP
        )
      : 0;

  const recall =
    TP + FN
      ? TP /
        (
          TP + FN
        )
      : 0;

  const specificity =
    TN + FP
      ? TN /
        (
          TN + FP
        )
      : 0;

  const FPR =
    FP + TN
      ? FP /
        (
          FP + TN
        )
      : 0;

  const FNR =
    FN + TP
      ? FN /
        (
          FN + TP
        )
      : 0;

  /*
    Overall safe accuracy:

    Abstentions are not counted as
    wrong classifications.
    They are reported separately.
  */

  return {

    total,

    decided,

    abstained,

    abstainAI,

    abstainHuman,

    TP,
    TN,
    FP,
    FN,

    coverage:
      Math.round(
        coverage *
        100
      ),

    selectiveAccuracy:
      Math.round(
        selectiveAccuracy *
        100
      ),

    precision:
      Math.round(
        precision *
        100
      ),

    recall:
      Math.round(
        recall *
        100
      ),

    specificity:
      Math.round(
        specificity *
        100
      ),

    falsePositiveRate:
      Math.round(
        FPR *
        100
      ),

    falseNegativeRate:
      Math.round(
        FNR *
        100
      )
  };
}

/* =========================================================
   BENCHMARK PROMPT
========================================================= */

function saveGroundTruth(
  scan
) {

  setTimeout(
    () => {

      const answer =
        prompt(
`AI TRACE V4.4 BENCHMARK

Do you KNOW the true origin?

Type:
AI
or
HUMAN

Cancel / blank = skip.`
        );

      if (!answer) {
        return;
      }

      const truth =
        answer
          .trim()
          .toUpperCase();

      if (
        truth !== 'AI' &&
        truth !== 'HUMAN'
      ) {

        alert(
          'Use only AI or HUMAN.'
        );

        return;
      }

      const records =
        loadBenchmark();

      const id =
        nextID(
          truth,
          records
        );

      const source =
        prompt(
          'Source / note:',
          truth === 'AI'
            ? 'Known AI sample'
            : 'Known human sample'
        ) || '';

      records.push({

        id,

        version:
          VERSION,

        timestamp:
          new Date()
            .toISOString(),

        groundTruth:
          truth,

        source,

        ...scan
      });

      saveBenchmark(
        records
      );

      const m =
        benchmarkMetrics();

      alert(
`Saved ${id}

Total: ${m.total}

DECIDED
TP: ${m.TP}
TN: ${m.TN}
FP: ${m.FP}
FN: ${m.FN}

ABSTAINED
AI: ${m.abstainAI}
Human: ${m.abstainHuman}

Coverage: ${m.coverage}%

Selective accuracy:
${m.selectiveAccuracy}%

Precision:
${m.precision}%

Recall:
${m.recall}%

Specificity:
${m.specificity}%

False Positive Rate:
${m.falsePositiveRate}%

False Negative Rate:
${m.falseNegativeRate}%`
      );

      renderBenchmarkDashboard();

    },
    700
  );
}

/* =========================================================
   HISTORY
========================================================= */

function saveHistory(
  record
) {

  try {

    const history =
      JSON.parse(
        localStorage.getItem(
          HISTORY_KEY
        ) || '[]'
      );

    history.push(
      record
    );

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(
        history.slice(-100)
      )
    );

  } catch (
    error
  ) {

    console.warn(
      error
    );
  }
}

/* =========================================================
   MAIN SCAN
========================================================= */

async function run() {

  const value =
    text.value.trim();

  const wordCount =
    value
      ? value
          .split(/\s+/)
          .filter(Boolean)
          .length
      : 0;

  if (
    wordCount < 80
  ) {

    alert(
      'Use at least 80 words.'
    );

    return;
  }

  $('scan').disabled =
    true;

  progress(
    3,
    'Building profile…'
  );

  const profile =
    createProfile(
      value
    );

  const language =
    detectLanguage(
      value
    );

  const chunks =
    chunkText(
      value
    );

  let tmr =
    50;

  let modern =
    50;

  let tmrWorked =
    true;

  let modernWorked =
    true;

  const segmentScores =
    [];

  /* =====================
     MODEL A
  ===================== */

  try {

    const modelA =
      await loadTMR();

    tmr =
      await classify(
        modelA,
        value
      );

    for (
      let i = 0;
      i < chunks.length;
      i++
    ) {

      progress(
        20 +
        Math.round(
          (
            i /
            chunks.length
          ) *
          30
        ),
        `Analyzing segment ${i + 1}/${chunks.length}`
      );

      segmentScores.push(
        await classify(
          modelA,
          chunks[i]
        )
      );
    }

  } catch (
    error
  ) {

    console.error(
      error
    );

    tmrWorked =
      false;

    segmentScores.push(
      50
    );
  }

  /* =====================
     MODEL B
  ===================== */

  try {

    const modelB =
      await loadModern();

    progress(
      70,
      'Deep analysis…'
    );

    modern =
      await classify(
        modelB,
        value
      );

  } catch (
    error
  ) {

    console.error(
      error
    );

    modernWorked =
      false;
  }

  /* =====================
     ENGINE
  ===================== */

  progress(
    88,
    'Calibrating evidence…'
  );

  const raw =
    rawEnsemble(
      tmr,
      modern
    );

  const modelGap =
    Math.abs(
      tmr -
      modern
    );

  const segmentDeviation =
    Math.round(
      std(
        segmentScores
      )
    );

  const segmentRange =
    segmentScores.length
      ? Math.max(
          ...segmentScores
        ) -
        Math.min(
          ...segmentScores
        )
      : 0;

  const modelsActive =
    Number(
      tmrWorked
    ) +
    Number(
      modernWorked
    );

  const evidenceQuality =
    calculateEvidenceQuality({

      modelGap,

      segmentDeviation,

      segmentRange,

      words:
        profile.words,

      language,

      modelsActive
    });

  const calibrated =
    calibrate(
      raw,
      evidenceQuality
    );

  const verdict =
    verdictV44({

      calibrated,

      evidenceQuality,

      modelGap,

      segmentRange,

      segmentDeviation,

      modelsActive
    });

  const uncertainty =
    100 -
    evidenceQuality;

  const result = {

    profile,

    language,

    tmr,

    modern,

    raw,

    calibrated,

    evidenceQuality,

    uncertainty,

    verdict,

    modelGap,

    segmentDeviation,

    segmentRange,

    modelsActive,

    segmentScores,

    chunks
  };

  renderResult(
    result
  );

  saveHistory({

    timestamp:
      new Date()
        .toISOString(),

    version:
      VERSION,

    words:
      profile.words,

    language,

    tmr,

    modern,

    raw,

    calibrated,

    evidenceQuality,

    verdict,

    modelGap,

    segmentDeviation,

    segmentRange
  });

  progress(
    100,
    'Complete'
  );

  $('modelState')
    .textContent =
      `V4.4 FP Defense ✓`;

  $('scan').disabled =
    false;

  setTimeout(
    () => {

      $('progress')
        ?.classList
        .add('hidden');

    },
    400
  );

  saveGroundTruth({

    words:
      profile.words,

    language,

    tmr,

    modern,

    raw,

    calibrated,

    evidenceQuality,

    uncertainty,

    verdict,

    modelGap,

    segmentDeviation,

    segmentRange,

    segmentScores
  });
}

/* =========================================================
   RESULT RENDERING
========================================================= */

function escapeHTML(
  value
) {

  return value.replace(
    /[&<>"']/g,
    c =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      }[c])
  );
}

function renderResult(r) {

  $('report')
    .classList
    .remove('hidden');

  /*
    Main display = calibrated score,
    NOT raw signal.
  */

  $('score')
    .textContent =
      r.calibrated +
      '%';

  $('scaleFill')
    .style.width =
      r.calibrated +
      '%';

  $('verdict')
    .textContent =
      r.verdict;

  const confidenceLabel =
    r.evidenceQuality >= 75
      ? 'High'
      : r.evidenceQuality >= 50
      ? 'Medium'
      : 'Low';

  $('confidence')
    .textContent =
      `Confidence: ${confidenceLabel} (${r.evidenceQuality}%)`;

  $('explain')
    .textContent =
`Raw detector signal: ${r.raw}%.
Calibrated score: ${r.calibrated}%.
Evidence quality: ${r.evidenceQuality}%.
TMR: ${r.tmr}%.
ModernBERT: ${r.modern}%.
Model disagreement: ${r.modelGap} points.
Segment range: ${r.segmentRange} points.`;

  const human =
    100 -
    r.calibrated;

  $('humanVal')
    .textContent =
      human + '%';

  $('aiVal')
    .textContent =
      r.calibrated + '%';

  $('uncertainVal')
    .textContent =
      r.uncertainty + '%';

  $('humanBar')
    .style.width =
      human + '%';

  $('aiBar')
    .style.width =
      r.calibrated + '%';

  $('uncertainBar')
    .style.width =
      r.uncertainty + '%';

  $('engineBadge')
    .textContent =
      'V4.4 • FP DEFENSE';

  const evidence = [

    [
      'Raw detector signal',
      `${r.raw}%`,
      'Raw'
    ],

    [
      'Calibrated score',
      `${r.calibrated}%`,
      'Adjusted'
    ],

    [
      'Evidence quality',
      `${r.evidenceQuality}%`,
      r.evidenceQuality >= 75
        ? 'Strong'
        : r.evidenceQuality >= 50
        ? 'Medium'
        : 'Weak'
    ],

    [
      'TMR detector',
      `${r.tmr}% AI signal`,
      'Model A'
    ],

    [
      'ModernBERT detector',
      `${r.modern}% AI signal`,
      'Model B'
    ],

    [
      'Model disagreement',
      `${r.modelGap} points`,
      r.modelGap >= 35
        ? 'Conflict'
        : 'Acceptable'
    ],

    [
      'Segment deviation',
      `${r.segmentDeviation}`,
      r.segmentDeviation > 28
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Segment range',
      `${r.segmentRange} points`,
      r.segmentRange > 60
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Language fit',
      r.language === 'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English — reliability reduced.',
      'Context'
    ]
  ];

  $('evidence')
    .innerHTML =
      evidence
        .map(
          item => `
<div class="ev">
  <div class="evTop">
    <span>${item[0]}</span>
    <span>${item[2]}</span>
  </div>
  <small>${item[1]}</small>
</div>`
        )
        .join('');

  const metrics = {

    Words:
      r.profile.words,

    Sentences:
      r.profile.sentences,

    'Avg. words / sentence':
      r.profile.avg.toFixed(1),

    'Lexical diversity':
      Math.round(
        r.profile.lexical *
        100
      ) + '%',

    Language:
      r.language,

    'Models active':
      `${r.modelsActive}/2`,

    'Raw signal':
      `${r.raw}%`,

    'Calibrated score':
      `${r.calibrated}%`,

    'Evidence quality':
      `${r.evidenceQuality}%`,

    'Model disagreement':
      `${r.modelGap} pts`,

    'Segment deviation':
      r.segmentDeviation,

    'Segment range':
      `${r.segmentRange} pts`
  };

  $('metrics')
    .innerHTML =
      Object
        .entries(metrics)
        .map(
          ([k, v]) => `
<div class="metric">
  <span>${k}</span>
  <b>${v}</b>
</div>`
        )
        .join('');

  $('segments')
    .innerHTML =
      r.chunks
        .map(
          (
            chunk,
            i
          ) => `
<div class="segment">

  <div class="segmentHead">

    <b>
      Segment ${i + 1}
    </b>

    <span>
      ${r.segmentScores[i] ?? 50}% TMR signal
    </span>

  </div>

  <div class="segmentMeter">

    <i style="width:${r.segmentScores[i] ?? 50}%"></i>

  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        300
      )
    )}

    ${
      chunk.length > 300
        ? '…'
        : ''
    }
  </p>

</div>`
        )
        .join('');

  $('report')
    .scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
}

/* =========================================================
   BENCHMARK DASHBOARD
========================================================= */

function renderBenchmarkDashboard() {

  const old =
    document.getElementById(
      'v44Benchmark'
    );

  if (old) {
    old.remove();
  }

  const records =
    loadBenchmark();

  if (!records.length) {
    return;
  }

  const metrics =
    benchmarkMetrics();

  const section =
    document.createElement(
      'section'
    );

  section.id =
    'v44Benchmark';

  section.className =
    'panel';

  section.style.marginTop =
    '18px';

  section.innerHTML = `
<span class="over">
  V4.4 BENCHMARK
</span>

<h2>
  False-Positive Defense
</h2>

<div class="metrics">

  <div class="metric">
    <span>Total samples</span>
    <b>${metrics.total}</b>
  </div>

  <div class="metric">
    <span>Decided</span>
    <b>${metrics.decided}</b>
  </div>

  <div class="metric">
    <span>Abstained</span>
    <b>${metrics.abstained}</b>
  </div>

  <div class="metric">
    <span>Coverage</span>
    <b>${metrics.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${metrics.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${metrics.precision}%</b>
  </div>

  <div class="metric">
    <span>Recall</span>
    <b>${metrics.recall}%</b>
  </div>

  <div class="metric">
    <span>Specificity</span>
    <b>${metrics.specificity}%</b>
  </div>

  <div class="metric">
    <span>False Positive Rate</span>
    <b>${metrics.falsePositiveRate}%</b>
  </div>

  <div class="metric">
    <span>False Negative Rate</span>
    <b>${metrics.falseNegativeRate}%</b>
  </div>

  <div class="metric">
    <span>Human abstentions</span>
    <b>${metrics.abstainHuman}</b>
  </div>

  <div class="metric">
    <span>AI abstentions</span>
    <b>${metrics.abstainAI}</b>
  </div>

</div>

<div class="notice" style="margin-top:16px">
  V4.4 treats INCONCLUSIVE as an abstention,
  not as a wrong Human/AI prediction.
  Accuracy is therefore reported only
  on samples where the engine actually
  makes a decision.
</div>
`;

  const report =
    $('report');

  report.parentNode
    .insertBefore(
      section,
      report.nextSibling
    );
}

/* =========================================================
   DEVELOPER UTILITIES
========================================================= */

window.AITraceV44 = {

  metrics() {
    return benchmarkMetrics();
  },

  records() {
    return loadBenchmark();
  },

  clearBenchmark() {

    if (
      confirm(
        'Delete all V4.4 benchmark data?'
      )
    ) {

      localStorage.removeItem(
        BENCHMARK_KEY
      );

      location.reload();
    }
  },

  exportBenchmark() {

    const payload = {

      version:
        VERSION,

      metrics:
        benchmarkMetrics(),

      records:
        loadBenchmark()
    };

    const blob =
      new Blob(
        [
          JSON.stringify(
            payload,
            null,
            2
          )
        ],
        {
          type:
            'application/json'
        }
      );

    const url =
      URL.createObjectURL(
        blob
      );

    const a =
      document.createElement(
        'a'
      );

    a.href =
      url;

    a.download =
      `AI-Trace-V44-Benchmark-${Date.now()}.json`;

    a.click();

    URL.revokeObjectURL(
      url
    );
  }
};

renderBenchmarkDashboard();
