/*
==========================================================
 AI TRACE V4.3.1
 Benchmark Dashboard Build

 ONE FILE REPLACEMENT

 - TMR detector
 - ModernBERT detector
 - 2-model analysis
 - calibration
 - evidence quality
 - segment stability
 - HUMAN / AI / MIXED benchmark
 - stored benchmark history
 - confusion matrix
 - false-positive / false-negative rates
 - experimental threshold search
 - visible BENCHMARK RESULTS button
 - no browser console required
==========================================================
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';


/* ======================================================
   CONFIG
====================================================== */

env.allowLocalModels = false;
env.useBrowserCache = true;

const VERSION = '4.3.1';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

/*
 IMPORTANT:
 Do not change this key.
 It is the same benchmark storage used by V4.3.
*/
const BENCHMARK_KEY =
  'aiTraceBenchmarkV43';

const HISTORY_KEY =
  'aiTraceHistoryV43';

let tmrClassifier = null;
let modernClassifier = null;


/* ======================================================
   DOM
====================================================== */

const $ = id =>
  document.getElementById(id);

const text = $('text');


/* ======================================================
   BENCHMARK DASHBOARD UI
====================================================== */

function installBenchmarkButton() {

  if (
    document.getElementById(
      'benchmarkResultsBtn'
    )
  ) {
    return;
  }

  const button =
    document.createElement(
      'button'
    );

  button.id =
    'benchmarkResultsBtn';

  button.textContent =
    'Benchmark Results';

  button.type =
    'button';

  button.style.cssText = `
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 9998;
    border: 1px solid #354764;
    background: #eaf1ff;
    color: #07101e;
    padding: 12px 16px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 12px;
    box-shadow: 0 10px 35px rgba(0,0,0,.35);
    cursor: pointer;
  `;

  button.onclick =
    showBenchmarkDashboard;

  document.body.appendChild(
    button
  );
}


function installBenchmarkModal() {

  if (
    document.getElementById(
      'benchmarkModal'
    )
  ) {
    return;
  }

  const modal =
    document.createElement(
      'div'
    );

  modal.id =
    'benchmarkModal';

  modal.style.cssText = `
    display: none;
    position: fixed;
    inset: 0;
    z-index: 9999;
    background: rgba(3,7,14,.88);
    padding: 18px;
    overflow-y: auto;
    backdrop-filter: blur(8px);
  `;

  modal.innerHTML = `
    <div
      style="
        max-width:760px;
        margin:30px auto;
        background:#0b1423;
        color:#eef3ff;
        border:1px solid #273650;
        border-radius:20px;
        padding:20px;
        box-shadow:0 25px 80px rgba(0,0,0,.5);
      "
    >

      <div
        style="
          display:flex;
          justify-content:space-between;
          align-items:flex-start;
          gap:15px;
          margin-bottom:18px;
        "
      >

        <div>
          <div
            style="
              color:#7187aa;
              font-size:10px;
              font-weight:800;
              letter-spacing:.15em;
            "
          >
            AI TRACE RESEARCH
          </div>

          <h2
            style="
              margin:6px 0 0;
              font-size:24px;
            "
          >
            Benchmark Results
          </h2>

          <div
            style="
              margin-top:5px;
              color:#7c90ad;
              font-size:12px;
            "
          >
            V4.3.1 Calibration Dashboard
          </div>
        </div>

        <button
          id="closeBenchmarkModal"
          style="
            border:0;
            background:#17253b;
            color:#dce5f3;
            border-radius:10px;
            padding:9px 12px;
            cursor:pointer;
          "
        >
          Close
        </button>

      </div>

      <div id="benchmarkDashboardContent"></div>

    </div>
  `;

  document.body.appendChild(
    modal
  );

  document
    .getElementById(
      'closeBenchmarkModal'
    )
    .onclick =
      hideBenchmarkDashboard;

  modal.addEventListener(
    'click',
    event => {

      if (
        event.target === modal
      ) {
        hideBenchmarkDashboard();
      }
    }
  );
}


function showBenchmarkDashboard() {

  installBenchmarkModal();

  const modal =
    document.getElementById(
      'benchmarkModal'
    );

  const content =
    document.getElementById(
      'benchmarkDashboardContent'
    );

  const records =
    loadBenchmark();

  const metrics =
    benchmarkMetrics(
      records
    );

  const thresholds =
    searchThresholds(
      records
    );

  const aiCount =
    records.filter(
      record =>
        record.groundTruth ===
        'AI'
    ).length;

  const humanCount =
    records.filter(
      record =>
        record.groundTruth ===
        'HUMAN'
    ).length;

  const mixedCount =
    records.filter(
      record =>
        record.groundTruth ===
        'MIXED'
    ).length;


  const card = (
    title,
    value,
    note = ''
  ) => `
    <div
      style="
        background:#07101c;
        border:1px solid #1e2d45;
        border-radius:13px;
        padding:14px;
      "
    >
      <div
        style="
          font-size:10px;
          letter-spacing:.08em;
          color:#7086a6;
        "
      >
        ${title}
      </div>

      <div
        style="
          font-size:25px;
          font-weight:900;
          margin-top:5px;
        "
      >
        ${value}
      </div>

      ${
        note
          ? `
            <div
              style="
                margin-top:5px;
                font-size:10px;
                color:#667b99;
              "
            >
              ${note}
            </div>
          `
          : ''
      }
    </div>
  `;


  const qualityLabel =
    metrics.samples < 20
      ? 'VERY EARLY DATA'
      : metrics.samples < 100
        ? 'EARLY BENCHMARK'
        : 'BENCHMARK DATA';


  content.innerHTML = `

    <div
      style="
        padding:13px 14px;
        background:#101d31;
        border:1px solid #263a58;
        border-radius:13px;
        color:#9db0ca;
        line-height:1.55;
        font-size:12px;
        margin-bottom:16px;
      "
    >
      <strong
        style="color:#dbe5f3"
      >
        ${qualityLabel}
      </strong>

      <br>

      These metrics are experimental.
      A small benchmark cannot establish production accuracy.
    </div>


    <div
      style="
        display:grid;
        grid-template-columns:repeat(2,minmax(0,1fr));
        gap:10px;
      "
    >

      ${card(
        'TOTAL RECORDS',
        records.length
      )}

      ${card(
        'BINARY SAMPLES',
        metrics.samples,
        'AI + HUMAN'
      )}

      ${card(
        'AI',
        aiCount
      )}

      ${card(
        'HUMAN',
        humanCount
      )}

      ${card(
        'MIXED',
        mixedCount
      )}

      ${card(
        'UNCERTAIN',
        metrics.uncertain
      )}

    </div>


    <h3
      style="
        margin:24px 0 12px;
      "
    >
      Confusion Matrix
    </h3>


    <div
      style="
        display:grid;
        grid-template-columns:repeat(2,minmax(0,1fr));
        gap:10px;
      "
    >

      ${card(
        'TRUE POSITIVE',
        metrics.TP,
        'AI → AI'
      )}

      ${card(
        'TRUE NEGATIVE',
        metrics.TN,
        'Human → Human'
      )}

      ${card(
        'FALSE POSITIVE',
        metrics.FP,
        'Human → AI'
      )}

      ${card(
        'FALSE NEGATIVE',
        metrics.FN,
        'AI → Human'
      )}

    </div>


    <h3
      style="
        margin:24px 0 12px;
      "
    >
      Performance
    </h3>


    <div
      style="
        display:grid;
        grid-template-columns:repeat(2,minmax(0,1fr));
        gap:10px;
      "
    >

      ${card(
        'ACCURACY',
        metrics.accuracy + '%'
      )}

      ${card(
        'PRECISION',
        metrics.precision + '%'
      )}

      ${card(
        'RECALL',
        metrics.recall + '%'
      )}

      ${card(
        'SPECIFICITY',
        metrics.specificity + '%'
      )}

      ${card(
        'FALSE POSITIVE RATE',
        metrics.falsePositiveRate + '%',
        'Critical metric'
      )}

      ${card(
        'FALSE NEGATIVE RATE',
        metrics.falseNegativeRate + '%'
      )}

    </div>


    <h3
      style="
        margin:24px 0 12px;
      "
    >
      Experimental Calibration
    </h3>


    ${
      thresholds
        ? `
          <div
            style="
              background:#07101c;
              border:1px solid #1e2d45;
              border-radius:13px;
              padding:16px;
              line-height:1.7;
              font-size:13px;
            "
          >

            Suggested HUMAN threshold:
            <strong>
              ≤ ${thresholds.humanThreshold}%
            </strong>

            <br>

            Suggested AI threshold:
            <strong>
              ≥ ${thresholds.aiThreshold}%
            </strong>

            <br><br>

            <span
              style="
                color:#7186a4;
                font-size:11px;
              "
            >
              Research-only recommendation.
              It is not automatically applied to production classification.
            </span>

          </div>
        `
        : `
          <div
            style="
              background:#07101c;
              border:1px solid #1e2d45;
              border-radius:13px;
              padding:16px;
              color:#8297b5;
              font-size:12px;
              line-height:1.6;
            "
          >
            At least 10 HUMAN and 10 AI samples
            are required before threshold search begins.
          </div>
        `
    }


    <h3
      style="
        margin:24px 0 12px;
      "
    >
      Benchmark Records
    </h3>


    <div
      style="
        display:grid;
        gap:8px;
      "
    >

      ${
        records.length
          ? records
              .slice()
              .reverse()
              .map(
                record => `
                  <div
                    style="
                      background:#07101c;
                      border:1px solid #1e2d45;
                      border-radius:12px;
                      padding:12px;
                    "
                  >

                    <div
                      style="
                        display:flex;
                        justify-content:space-between;
                        gap:10px;
                        font-size:12px;
                      "
                    >

                      <strong>
                        ${record.id || 'Sample'}
                      </strong>

                      <span
                        style="color:#91a5c2"
                      >
                        ${record.groundTruth}
                      </span>

                    </div>

                    <div
                      style="
                        margin-top:6px;
                        color:#7186a5;
                        font-size:10px;
                        line-height:1.5;
                      "
                    >

                      Calibrated:
                      ${record.calibratedScore ?? '--'}%

                      · Evidence:
                      ${record.evidenceQuality ?? '--'}%

                      · TMR:
                      ${record.tmr ?? '--'}%

                      · Modern:
                      ${record.modern ?? '--'}%

                    </div>

                  </div>
                `
              )
              .join('')
          : `
            <div
              style="
                color:#7489a8;
                font-size:12px;
              "
            >
              No benchmark data found on this device/browser.
            </div>
          `
      }

    </div>


    <div
      style="
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin-top:22px;
      "
    >

      <button
        id="exportBenchmarkBtn"
        style="
          border:0;
          background:#eaf1ff;
          color:#07101e;
          padding:11px 14px;
          border-radius:10px;
          font-weight:800;
          cursor:pointer;
        "
      >
        Export JSON
      </button>

      <button
        id="clearBenchmarkBtn"
        style="
          border:1px solid #3a2b34;
          background:#20141a;
          color:#d2aebc;
          padding:11px 14px;
          border-radius:10px;
          font-weight:800;
          cursor:pointer;
        "
      >
        Clear Benchmark
      </button>

    </div>
  `;


  modal.style.display =
    'block';


  document
    .getElementById(
      'exportBenchmarkBtn'
    )
    .onclick =
      exportBenchmarkJSON;


  document
    .getElementById(
      'clearBenchmarkBtn'
    )
    .onclick = () => {

      const confirmation =
        confirm(
          'Delete ALL benchmark results stored in this browser?'
        );

      if (
        !confirmation
      ) {
        return;
      }

      localStorage.removeItem(
        BENCHMARK_KEY
      );

      showBenchmarkDashboard();
    };
}


function hideBenchmarkDashboard() {

  const modal =
    document.getElementById(
      'benchmarkModal'
    );

  if (
    modal
  ) {
    modal.style.display =
      'none';
  }
}


/* ======================================================
   INPUT EVENTS
====================================================== */

text.oninput = () => {

  const wordCount =
    text.value.trim()
      ? text.value
          .trim()
          .split(/\s+/)
          .filter(Boolean)
          .length
      : 0;

  $('count')
    .textContent =
      `${wordCount} words`;
};


$('clear').onclick = () => {

  text.value = '';

  text.oninput();

  $('report')
    .classList
    .add('hidden');
};


$('demo').onclick = () => {

  text.value =
`Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

Organizations can use artificial intelligence to identify patterns, automate repetitive processes, and support decision-making. However, automated systems can also produce incorrect or misleading information, which means human oversight remains important.

As these technologies become more capable, transparency about how digital content was created or modified will become increasingly valuable. Verification systems may therefore play an important role in establishing trust across the digital environment.`;

  text.oninput();
};


$('scan').onclick =
  run;


/* ======================================================
   PROGRESS
====================================================== */

function progress(
  percent,
  message
) {

  $('progress')
    .classList
    .remove('hidden');

  $('bar')
    .style
    .width =
      percent + '%';

  $('progressText')
    .textContent =
      message;
}


/* ======================================================
   HELPERS
====================================================== */

function mean(
  values
) {

  if (
    !values.length
  ) {
    return 0;
  }

  return (
    values.reduce(
      (a, b) =>
        a + b,
      0
    ) /
    values.length
  );
}


function std(
  values
) {

  if (
    !values.length
  ) {
    return 0;
  }

  const average =
    mean(values);

  return Math.sqrt(
    mean(
      values.map(
        value =>
          (
            value -
            average
          ) ** 2
      )
    )
  );
}


function clamp(
  value,
  min,
  max
) {

  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );
}


/* ======================================================
   LANGUAGE
====================================================== */

function detectLanguage(
  value
) {

  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) || []
    ).length;

  const total =
    (
      value.match(
        /\p{L}/gu
      ) || []
    ).length;

  if (!total) {
    return 'Unknown';
  }

  return (
    latin /
    total >
    0.8
  )
    ? 'English'
    : 'Non-English';
}


/* ======================================================
   DOCUMENT PROFILE
====================================================== */

function createProfile(
  value
) {

  const words =
    value
      .trim()
      .split(/\s+/)
      .filter(Boolean);

  const sentences =
    value
      .split(/[.!?]+/)
      .map(
        sentence =>
          sentence.trim()
      )
      .filter(Boolean);

  const cleanedWords =
    words
      .map(
        word =>
          word
            .toLowerCase()
            .replace(
              /[^\p{L}\p{N}]/gu,
              ''
            )
      )
      .filter(Boolean);

  const avgSentence =
    words.length /
    Math.max(
      1,
      sentences.length
    );

  const lexicalDiversity =
    new Set(
      cleanedWords
    ).size /
    Math.max(
      1,
      cleanedWords.length
    );

  const lengths =
    sentences.map(
      sentence =>
        sentence
          .split(/\s+/)
          .filter(Boolean)
          .length
    );

  const averageLength =
    mean(lengths);

  const variance =
    lengths.length
      ? mean(
          lengths.map(
            length =>
              (
                length -
                averageLength
              ) ** 2
          )
        )
      : 0;

  return {
    words:
      words.length,

    sentences:
      sentences.length,

    avgSentence,

    lexicalDiversity,

    variance
  };
}


/* ======================================================
   CHUNKING
====================================================== */

function chunkText(
  value,
  maxCharacters = 1450
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
        maxCharacters &&
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

  if (
    current.trim()
  ) {

    chunks.push(
      current.trim()
    );
  }

  return chunks
    .filter(Boolean)
    .slice(
      0,
      8
    );
}


/* ======================================================
   MODEL LOADING
====================================================== */

async function loadTMR() {

  if (
    tmrClassifier
  ) {
    return tmrClassifier;
  }

  $('modelState')
    .textContent =
      'Loading TMR…';

  progress(
    10,
    'Loading detector A…'
  );

  tmrClassifier =
    await pipeline(
      'text-classification',
      TMR_MODEL,
      {
        dtype:
          'q8'
      }
    );

  return tmrClassifier;
}


async function loadModern() {

  if (
    modernClassifier
  ) {
    return modernClassifier;
  }

  $('modelState')
    .textContent =
      'Loading ModernBERT…';

  progress(
    55,
    'Loading detector B…'
  );

  modernClassifier =
    await pipeline(
      'text-classification',
      MODERN_MODEL,
      {
        dtype:
          'q4f16'
      }
    );

  return modernClassifier;
}


/* ======================================================
   MODEL OUTPUT PARSER
====================================================== */

function getAIProbability(
  output
) {

  const results =
    (
      Array.isArray(output)
        ? output
        : [output]
    ).flat();

  let ai = null;
  let human = null;

  for (
    const result
    of results
  ) {

    const label =
      String(
        result.label || ''
      )
        .toLowerCase();

    const score =
      Number(
        result.score
      ) || 0;

    if (
      label === 'ai' ||
      label.includes(
        'machine'
      ) ||
      label.includes(
        'generated'
      ) ||
      label ===
        'label_1'
    ) {

      ai =
        Math.max(
          ai ?? 0,
          score
        );
    }

    if (
      label ===
        'human' ||
      label ===
        'label_0'
    ) {

      human =
        Math.max(
          human ?? 0,
          score
        );
    }
  }

  if (
    ai !== null
  ) {
    return ai;
  }

  if (
    human !== null
  ) {
    return (
      1 -
      human
    );
  }

  if (
    results.length >= 2
  ) {

    return Number(
      results[1]
        ?.score ??
      0.5
    );
  }

  return 0.5;
}


async function classify(
  classifier,
  value
) {

  const result =
    await classifier(
      value,
      {
        top_k: null,
        truncation: true
      }
    );

  return Math.round(
    getAIProbability(
      result
    ) *
    100
  );
}


/* ======================================================
   EVIDENCE QUALITY
====================================================== */

function calculateEvidenceQuality({
  tmr,
  modern,
  segments,
  language,
  words,
  bothModels
}) {

  let quality =
    100;

  if (
    !bothModels
  ) {
    quality -= 40;
  }

  const modelGap =
    Math.abs(
      tmr -
      modern
    );

  quality -=
    Math.min(
      25,
      modelGap *
        0.8
    );

  const deviation =
    std(
      segments
    );

  quality -=
    Math.min(
      35,
      deviation *
        0.85
    );

  const segmentRange =
    segments.length
      ? (
          Math.max(
            ...segments
          ) -
          Math.min(
            ...segments
          )
        )
      : 0;

  if (
    segmentRange >= 70
  ) {
    quality -= 15;
  }

  if (
    words < 150
  ) {
    quality -= 10;
  }

  if (
    language !==
      'English'
  ) {
    quality -= 25;
  }

  return clamp(
    Math.round(
      quality
    ),
    5,
    100
  );
}


/* ======================================================
   CALIBRATION
====================================================== */

function calibrateScore(
  rawSignal,
  evidenceQuality
) {

  const quality =
    evidenceQuality /
    100;

  return Math.round(
    50 +
    (
      rawSignal -
      50
    ) *
    quality
  );
}


/* ======================================================
   VERDICT
====================================================== */

function verdictFor({
  calibratedScore,
  evidenceQuality,
  segmentRange,
  modelGap
}) {

  if (
    evidenceQuality < 55
  ) {
    return 'INCONCLUSIVE';
  }

  if (
    segmentRange >= 70
  ) {
    return 'INCONCLUSIVE';
  }

  if (
    modelGap >= 35
  ) {
    return 'INCONCLUSIVE';
  }

  if (
    calibratedScore >= 85
  ) {
    return 'Strong AI evidence';
  }

  if (
    calibratedScore >= 68
  ) {
    return 'Likely AI';
  }

  if (
    calibratedScore <= 15
  ) {
    return 'Strong human evidence';
  }

  if (
    calibratedScore <= 32
  ) {
    return 'Likely human';
  }

  return 'INCONCLUSIVE';
}


/* ======================================================
   BENCHMARK STORAGE
====================================================== */

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
  records
) {

  localStorage.setItem(
    BENCHMARK_KEY,
    JSON.stringify(
      records
    )
  );
}


/* ======================================================
   BENCHMARK IDS
====================================================== */

function nextID(
  truth,
  records
) {

  const prefix =
    truth === 'AI'
      ? 'A'
      : truth === 'HUMAN'
        ? 'H'
        : 'M';

  const count =
    records.filter(
      record =>
        record
          .groundTruth ===
        truth
    ).length + 1;

  return (
    prefix +
    '-' +
    String(
      count
    ).padStart(
      3,
      '0'
    )
  );
}


/* ======================================================
   BENCHMARK PREDICTION
====================================================== */

function benchmarkPrediction(
  record,
  aiThreshold = 68,
  humanThreshold = 32
) {

  if (
    record.evidenceQuality <
      55
  ) {
    return 'UNCERTAIN';
  }

  if (
    record.segmentRange >=
      70
  ) {
    return 'UNCERTAIN';
  }

  if (
    record.calibratedScore >=
      aiThreshold
  ) {
    return 'AI';
  }

  if (
    record.calibratedScore <=
      humanThreshold
  ) {
    return 'HUMAN';
  }

  return 'UNCERTAIN';
}


/* ======================================================
   BENCHMARK METRICS
====================================================== */

function benchmarkMetrics(
  records,
  aiThreshold = 68,
  humanThreshold = 32
) {

  const binary =
    records.filter(
      record =>
        record.groundTruth ===
          'AI' ||
        record.groundTruth ===
          'HUMAN'
    );

  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;
  let uncertain = 0;

  for (
    const record
    of binary
  ) {

    const prediction =
      benchmarkPrediction(
        record,
        aiThreshold,
        humanThreshold
      );

    if (
      prediction ===
        'UNCERTAIN'
    ) {

      uncertain++;
      continue;
    }

    if (
      record.groundTruth ===
        'AI' &&
      prediction ===
        'AI'
    ) {
      TP++;
    }

    if (
      record.groundTruth ===
        'HUMAN' &&
      prediction ===
        'HUMAN'
    ) {
      TN++;
    }

    if (
      record.groundTruth ===
        'HUMAN' &&
      prediction ===
        'AI'
    ) {
      FP++;
    }

    if (
      record.groundTruth ===
        'AI' &&
      prediction ===
        'HUMAN'
    ) {
      FN++;
    }
  }

  const decided =
    TP +
    TN +
    FP +
    FN;

  const accuracy =
    decided
      ? (
          TP +
          TN
        ) /
        decided
      : 0;

  const precision =
    (
      TP +
      FP
    )
      ? TP /
        (
          TP +
          FP
        )
      : 0;

  const recall =
    (
      TP +
      FN
    )
      ? TP /
        (
          TP +
          FN
        )
      : 0;

  const specificity =
    (
      TN +
      FP
    )
      ? TN /
        (
          TN +
          FP
        )
      : 0;

  const fpr =
    (
      FP +
      TN
    )
      ? FP /
        (
          FP +
          TN
        )
      : 0;

  const fnr =
    (
      FN +
      TP
    )
      ? FN /
        (
          FN +
          TP
        )
      : 0;

  return {

    samples:
      binary.length,

    decided,

    uncertain,

    TP,
    TN,
    FP,
    FN,

    accuracy:
      Math.round(
        accuracy *
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
        fpr *
        100
      ),

    falseNegativeRate:
      Math.round(
        fnr *
        100
      )
  };
}


/* ======================================================
   THRESHOLD SEARCH
====================================================== */

function searchThresholds(
  records
) {

  const humans =
    records.filter(
      record =>
        record.groundTruth ===
        'HUMAN'
    ).length;

  const ais =
    records.filter(
      record =>
        record.groundTruth ===
        'AI'
    ).length;

  if (
    humans < 10 ||
    ais < 10
  ) {
    return null;
  }

  let best = null;

  for (
    let humanThreshold = 15;
    humanThreshold <= 45;
    humanThreshold += 2
  ) {

    for (
      let aiThreshold = 55;
      aiThreshold <= 90;
      aiThreshold += 2
    ) {

      if (
        humanThreshold >=
        aiThreshold
      ) {
        continue;
      }

      const metrics =
        benchmarkMetrics(
          records,
          aiThreshold,
          humanThreshold
        );

      const score =
        metrics.accuracy
        -
        (
          metrics.falsePositiveRate *
          1.5
        )
        -
        (
          metrics.falseNegativeRate *
          0.5
        )
        -
        (
          (
            metrics.uncertain /
            Math.max(
              1,
              metrics.samples
            )
          ) *
          15
        );

      if (
        !best ||
        score >
          best.score
      ) {

        best = {

          humanThreshold,

          aiThreshold,

          score,

          metrics
        };
      }
    }
  }

  return best;
}


/* ======================================================
   RECORD BENCHMARK
====================================================== */

function recordBenchmark(
  data
) {

  const answer =
    prompt(
`AI TRACE BENCHMARK

Only label if the true origin is known.

AI
HUMAN
MIXED

Cancel = skip`
    );

  if (
    !answer
  ) {
    return;
  }

  const truth =
    answer
      .trim()
      .toUpperCase();

  if (
    ![
      'AI',
      'HUMAN',
      'MIXED'
    ].includes(
      truth
    )
  ) {

    alert(
      'Use AI, HUMAN or MIXED.'
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
      'Source / Note:',
      truth === 'AI'
        ? 'Known AI sample'
        : truth === 'HUMAN'
          ? 'Known human sample'
          : 'Known mixed sample'
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

    ...data
  });

  saveBenchmark(
    records
  );

  const metrics =
    benchmarkMetrics(
      records
    );

  alert(
`Saved ${id}

Binary samples: ${metrics.samples}

TP: ${metrics.TP}
TN: ${metrics.TN}
FP: ${metrics.FP}
FN: ${metrics.FN}
Uncertain: ${metrics.uncertain}

Accuracy: ${metrics.accuracy}%
Precision: ${metrics.precision}%
Recall: ${metrics.recall}%
Specificity: ${metrics.specificity}%

False Positive Rate: ${metrics.falsePositiveRate}%
False Negative Rate: ${metrics.falseNegativeRate}%`
  );
}


/* ======================================================
   EXPORT
====================================================== */

function exportBenchmarkJSON() {

  const records =
    loadBenchmark();

  const data =
    JSON.stringify(
      {
        version:
          VERSION,

        exportedAt:
          new Date()
            .toISOString(),

        metrics:
          benchmarkMetrics(
            records
          ),

        suggestedThresholds:
          searchThresholds(
            records
          ),

        records
      },
      null,
      2
    );

  const blob =
    new Blob(
      [data],
      {
        type:
          'application/json'
      }
    );

  const url =
    URL.createObjectURL(
      blob
    );

  const anchor =
    document.createElement(
      'a'
    );

  anchor.href =
    url;

  anchor.download =
    `AI-Trace-Benchmark-${Date.now()}.json`;

  anchor.click();

  URL.revokeObjectURL(
    url
  );
}


/* ======================================================
   HISTORY
====================================================== */

function saveHistory(
  result
) {

  try {

    const history =
      JSON.parse(
        localStorage.getItem(
          HISTORY_KEY
        ) || '[]'
      );

    history.push({

      timestamp:
        new Date()
          .toISOString(),

      ...result
    });

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(
        history.slice(
          -100
        )
      )
    );

  } catch {}
}


/* ======================================================
   MAIN ANALYSIS
====================================================== */

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
    'Building document profile…'
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

  let tmrScore = 50;
  let modernScore = 50;

  let tmrWorked = true;
  let modernWorked = true;

  const segmentScores =
    [];


  /* TMR */

  try {

    const tmr =
      await loadTMR();

    progress(
      18,
      'TMR document scan…'
    );

    tmrScore =
      await classify(
        tmr,
        value
      );

    for (
      let i = 0;
      i < chunks.length;
      i++
    ) {

      progress(
        22 +
        Math.round(
          (
            i /
            Math.max(
              1,
              chunks.length
            )
          ) *
          28
        ),
        `Segment ${i + 1}/${chunks.length}`
      );

      segmentScores.push(
        await classify(
          tmr,
          chunks[i]
        )
      );
    }

  } catch (
    error
  ) {

    console.error(
      'TMR failed',
      error
    );

    tmrWorked =
      false;

    segmentScores.push(
      50
    );
  }


  /* ModernBERT */

  try {

    const modern =
      await loadModern();

    progress(
      65,
      'ModernBERT Deep Scan…'
    );

    modernScore =
      await classify(
        modern,
        value
      );

  } catch (
    error
  ) {

    console.error(
      'ModernBERT failed',
      error
    );

    modernWorked =
      false;
  }


  /* Calibration */

  progress(
    85,
    'Calibrating evidence…'
  );

  let rawSignal;

  if (
    tmrWorked &&
    modernWorked
  ) {

    rawSignal =
      Math.round(
        (
          tmrScore +
          modernScore
        ) /
        2
      );

  } else if (
    tmrWorked
  ) {

    rawSignal =
      tmrScore;

  } else if (
    modernWorked
  ) {

    rawSignal =
      modernScore;

  } else {

    rawSignal =
      50;
  }

  const modelGap =
    (
      tmrWorked &&
      modernWorked
    )
      ? Math.abs(
          tmrScore -
          modernScore
        )
      : 100;

  const segmentDeviation =
    Math.round(
      std(
        segmentScores
      )
    );

  const segmentRange =
    segmentScores.length
      ? (
          Math.max(
            ...segmentScores
          ) -
          Math.min(
            ...segmentScores
          )
        )
      : 100;

  const evidenceQuality =
    calculateEvidenceQuality({

      tmr:
        tmrScore,

      modern:
        modernScore,

      segments:
        segmentScores,

      language,

      words:
        profile.words,

      bothModels:
        tmrWorked &&
        modernWorked
    });

  const calibratedScore =
    calibrateScore(
      rawSignal,
      evidenceQuality
    );

  const verdict =
    verdictFor({

      calibratedScore,

      evidenceQuality,

      segmentRange,

      modelGap
    });

  const confidence =
    evidenceQuality;

  const uncertainty =
    100 -
    evidenceQuality;


  render({

    profile,

    language,

    chunks,

    segmentScores,

    tmrScore,

    modernScore,

    tmrWorked,

    modernWorked,

    rawSignal,

    calibratedScore,

    evidenceQuality,

    confidence,

    uncertainty,

    modelGap,

    segmentDeviation,

    segmentRange,

    verdict
  });


  saveHistory({

    version:
      VERSION,

    words:
      profile.words,

    language,

    tmr:
      tmrScore,

    modern:
      modernScore,

    rawSignal,

    calibratedScore,

    evidenceQuality,

    verdict,

    modelGap,

    segmentDeviation,

    segmentRange
  });


  progress(
    100,
    'Trace complete'
  );

  $('modelState')
    .textContent =
      `V${VERSION} Benchmark Engine ✓`;

  setTimeout(
    () => {

      $('progress')
        .classList
        .add('hidden');

    },
    450
  );

  $('scan').disabled =
    false;


  setTimeout(
    () => {

      recordBenchmark({

        words:
          profile.words,

        language,

        tmr:
          tmrScore,

        modern:
          modernScore,

        rawSignal,

        calibratedScore,

        evidenceQuality,

        verdict,

        modelGap,

        segmentDeviation,

        segmentRange,

        segmentScores
      });

    },
    700
  );
}


/* ======================================================
   HTML ESCAPE
====================================================== */

function escapeHTML(
  value
) {

  return value.replace(
    /[&<>"']/g,
    character => (
      {
        '&':
          '&amp;',

        '<':
          '&lt;',

        '>':
          '&gt;',

        '"':
          '&quot;',

        "'":
          '&#039;'
      }
    )[character]
  );
}


/* ======================================================
   RENDER
====================================================== */

function render(
  result
) {

  const {

    profile,
    language,
    chunks,
    segmentScores,

    tmrScore,
    modernScore,

    tmrWorked,
    modernWorked,

    rawSignal,
    calibratedScore,

    evidenceQuality,
    confidence,
    uncertainty,

    modelGap,
    segmentDeviation,
    segmentRange,

    verdict

  } = result;


  $('report')
    .classList
    .remove('hidden');


  $('score')
    .textContent =
      calibratedScore +
      '%';

  $('scaleFill')
    .style
    .width =
      calibratedScore +
      '%';

  $('verdict')
    .textContent =
      verdict;


  let confidenceText =
    'Low';

  if (
    confidence >= 80
  ) {

    confidenceText =
      'High';

  } else if (
    confidence >= 55
  ) {

    confidenceText =
      'Medium';
  }


  $('confidence')
    .textContent =
      `Confidence: ${confidenceText} (${confidence}%)`;


  $('explain')
    .textContent =
`Raw detector signal: ${rawSignal}%.
Calibrated score: ${calibratedScore}%.
Evidence quality: ${evidenceQuality}%.
TMR: ${tmrScore}%.
ModernBERT: ${modernScore}%.
Model disagreement: ${modelGap} points.
Segment range: ${segmentRange} points.`;


  const humanSignal =
    100 -
    calibratedScore;


  $('humanVal')
    .textContent =
      humanSignal +
      '%';

  $('aiVal')
    .textContent =
      calibratedScore +
      '%';

  $('uncertainVal')
    .textContent =
      uncertainty +
      '%';


  $('humanBar')
    .style
    .width =
      humanSignal +
      '%';

  $('aiBar')
    .style
    .width =
      calibratedScore +
      '%';

  $('uncertainBar')
    .style
    .width =
      uncertainty +
      '%';


  $('engineBadge')
    .textContent =
      'V4.3.1 • BENCHMARK CALIBRATION';


  const evidence = [

    [
      'Raw detector signal',
      `${rawSignal}%`,
      'Raw'
    ],

    [
      'Calibrated score',
      `${calibratedScore}%`,
      'Adjusted'
    ],

    [
      'Evidence quality',
      `${evidenceQuality}%`,
      evidenceQuality >= 80
        ? 'Strong'
        : evidenceQuality >= 55
          ? 'Medium'
          : 'Weak'
    ],

    [
      'TMR detector',
      tmrWorked
        ? `${tmrScore}% AI signal`
        : 'Unavailable',
      'Model A'
    ],

    [
      'ModernBERT detector',
      modernWorked
        ? `${modernScore}% AI signal`
        : 'Unavailable',
      'Model B'
    ],

    [
      'Model disagreement',
      `${modelGap} points`,
      modelGap >= 35
        ? 'Conflict'
        : 'Acceptable'
    ],

    [
      'Segment deviation',
      `${segmentDeviation}`,
      segmentDeviation >= 28
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Segment range',
      `${segmentRange} points`,
      segmentRange >= 70
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Language fit',
      language === 'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English — reduced reliability.',
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

    <span>
      ${item[0]}
    </span>

    <span>
      ${item[2]}
    </span>

  </div>

  <small>
    ${item[1]}
  </small>

</div>`
        )
        .join('');


  const metrics = {

    Words:
      profile.words,

    Sentences:
      profile.sentences,

    'Avg. words / sentence':
      profile
        .avgSentence
        .toFixed(1),

    'Lexical diversity':
      Math.round(
        profile
          .lexicalDiversity *
        100
      ) + '%',

    Language:
      language,

    'Models active':
      `${
        Number(
          tmrWorked
        ) +
        Number(
          modernWorked
        )
      }/2`,

    'Raw signal':
      rawSignal + '%',

    'Calibrated score':
      calibratedScore + '%',

    'Evidence quality':
      evidenceQuality + '%',

    'Model disagreement':
      modelGap + ' pts',

    'Segment deviation':
      segmentDeviation,

    'Segment range':
      segmentRange + ' pts'
  };


  $('metrics')
    .innerHTML =
      Object
        .entries(
          metrics
        )
        .map(
          (
            [
              name,
              value
            ]
          ) => `
<div class="metric">

  <span>
    ${name}
  </span>

  <b>
    ${value}
  </b>

</div>`
        )
        .join('');


  $('segments')
    .innerHTML =
      chunks
        .map(
          (
            chunk,
            index
          ) => {

            const score =
              segmentScores[
                index
              ] ?? 50;

            return `
<div class="segment">

  <div class="segmentHead">

    <b>
      Segment ${index + 1}
    </b>

    <span>
      ${score}% TMR signal
    </span>

  </div>

  <div class="segmentMeter">

    <i
      style="width:${score}%"
    ></i>

  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        300
      )
    )}

    ${
      chunk.length >
        300
        ? '…'
        : ''
    }
  </p>

</div>`;
          }
        )
        .join('');


  $('report')
    .scrollIntoView({
      behavior:
        'smooth',

      block:
        'start'
    });
}


/* ======================================================
   INITIALIZE
====================================================== */

installBenchmarkButton();
installBenchmarkModal();


/* ======================================================
   OPTIONAL DEVELOPER API
====================================================== */

window.AITrace = {

  benchmark() {

    const records =
      loadBenchmark();

    return {

      version:
        VERSION,

      total:
        records.length,

      records,

      metrics:
        benchmarkMetrics(
          records
        ),

      suggestedThresholds:
        searchThresholds(
          records
        )
    };
  },


  showBenchmark:
    showBenchmarkDashboard,


  exportBenchmark:
    exportBenchmarkJSON
};
