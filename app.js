/*
  AI TRACE V4.2
  Evidence-Adjusted Calibration Engine

  One-file replacement.
  No benchmark.js dependency required.
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const APP_VERSION = '4.2';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCHMARK_STORAGE =
  'aiTraceBenchmarkV42';

const HISTORY_STORAGE =
  'aiTraceHistoryV42';

let tmrClassifier = null;
let modernClassifier = null;

const $ = id =>
  document.getElementById(id);

const text = $('text');

/* =====================================================
   UI
===================================================== */

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
    .classList
    .add('hidden');
};

$('demo').onclick = () => {
  text.value =
`Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

One of the most important advantages of artificial intelligence is its ability to process large amounts of information quickly. Organizations can use AI-powered tools to identify patterns, automate repetitive processes, and support better decision-making.

However, the growing use of artificial intelligence also creates important challenges. AI-generated information can sometimes be inaccurate, misleading, or difficult to distinguish from content created by humans.

The future will therefore require more than simply developing increasingly powerful artificial intelligence systems. Society will also need technologies that provide transparency, verification, and evidence about how digital content was created or modified.`;

  text.oninput();
};

$('scan').onclick = run;

function progress(
  percent,
  label
) {
  $('progress')
    .classList
    .remove('hidden');

  $('bar')
    .style
    .width =
      `${percent}%`;

  $('progressText')
    .textContent =
      label;
}

/* =====================================================
   LANGUAGE
===================================================== */

function detectLanguage(value) {
  const latin =
    (
      value.match(/[A-Za-z]/g) ||
      []
    ).length;

  const letters =
    (
      value.match(/\p{L}/gu) ||
      []
    ).length;

  if (!letters) {
    return 'Unknown';
  }

  return (
    latin / letters
  ) > 0.8
    ? 'English'
    : 'Non-English';
}

/* =====================================================
   DOCUMENT PROFILE
===================================================== */

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
      .map(word =>
        word
          .toLowerCase()
          .replace(
            /[^\p{L}\p{N}]/gu,
            ''
          )
      )
      .filter(Boolean);

  const avgSentenceLength =
    words.length /
    Math.max(
      sentences.length,
      1
    );

  const lexicalDiversity =
    new Set(cleaned).size /
    Math.max(
      cleaned.length,
      1
    );

  const sentenceLengths =
    sentences.map(
      sentence =>
        sentence
          .split(/\s+/)
          .filter(Boolean)
          .length
    );

  const sentenceMean =
    sentenceLengths.reduce(
      (a, b) => a + b,
      0
    ) /
    Math.max(
      sentenceLengths.length,
      1
    );

  const sentenceVariance =
    sentenceLengths.reduce(
      (sum, length) =>
        sum +
        (
          length -
          sentenceMean
        ) ** 2,
      0
    ) /
    Math.max(
      sentenceLengths.length,
      1
    );

  return {
    words:
      words.length,

    sentences:
      sentences.length,

    avgSentenceLength,

    lexicalDiversity,

    sentenceVariance
  };
}

/* =====================================================
   CHUNKING
===================================================== */

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
    .slice(0, 8);
}

/* =====================================================
   MODELS
===================================================== */

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

/* =====================================================
   MODEL OUTPUT
===================================================== */

function extractAIProbability(
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
      ).toLowerCase();

    const score =
      Number(
        result.score
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

  if (
    ai !== null
  ) {
    return ai;
  }

  if (
    human !== null
  ) {
    return 1 - human;
  }

  if (
    results.length >= 2
  ) {
    return Number(
      results[1]?.score ??
      0.5
    );
  }

  return 0.5;
}

async function classify(
  classifier,
  value
) {
  const output =
    await classifier(
      value,
      {
        top_k: null,
        truncation: true
      }
    );

  return Math.round(
    extractAIProbability(
      output
    ) *
    100
  );
}

/* =====================================================
   STATS
===================================================== */

function average(
  values
) {
  if (
    !values.length
  ) {
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

function standardDeviation(
  values
) {
  if (
    !values.length
  ) {
    return 0;
  }

  const mean =
    average(values);

  return Math.sqrt(
    average(
      values.map(
        value =>
          (
            value -
            mean
          ) ** 2
      )
    )
  );
}

/* =====================================================
   RAW MODEL SIGNAL
===================================================== */

function calculateRawSignal(
  tmr,
  modern,
  tmrWorked,
  modernWorked
) {
  if (
    tmrWorked &&
    modernWorked
  ) {
    return Math.round(
      tmr * 0.5 +
      modern * 0.5
    );
  }

  if (
    tmrWorked
  ) {
    return tmr;
  }

  if (
    modernWorked
  ) {
    return modern;
  }

  return 50;
}

/* =====================================================
   EVIDENCE QUALITY

   0.00 = weak evidence
   1.00 = excellent evidence
===================================================== */

function calculateEvidenceQuality({
  modelGap,
  segmentDeviation,
  segmentRange,
  words,
  language,
  activeModels
}) {
  let quality = 1.0;

  /*
    Model disagreement penalty.
  */

  quality -=
    Math.min(
      0.35,
      modelGap /
      100 *
      0.75
    );

  /*
    Segment instability penalty.
  */

  quality -=
    Math.min(
      0.30,
      segmentDeviation /
      100 *
      0.75
    );

  quality -=
    Math.min(
      0.25,
      segmentRange /
      100 *
      0.40
    );

  /*
    Short-text penalty.
  */

  if (
    words < 100
  ) {
    quality -=
      0.25;
  } else if (
    words < 150
  ) {
    quality -=
      0.12;
  }

  /*
    Non-English penalty.
  */

  if (
    language !==
      'English'
  ) {
    quality -=
      0.25;
  }

  /*
    Missing-model penalty.
  */

  if (
    activeModels < 2
  ) {
    quality -=
      0.35;
  }

  return Math.max(
    0.05,
    Math.min(
      1,
      quality
    )
  );
}

/* =====================================================
   V4.2 CALIBRATION

   Key idea:
   Raw detector output is pulled toward 50
   when evidence quality is weak.

   Example:
   raw = 90
   quality = .25
   calibrated ≈ 60

   raw = 93
   quality = .95
   calibrated ≈ 91
===================================================== */

function calibrateScore(
  rawSignal,
  evidenceQuality
) {
  const distanceFromNeutral =
    rawSignal -
    50;

  const adjustedDistance =
    distanceFromNeutral *
    evidenceQuality;

  return Math.round(
    50 +
    adjustedDistance
  );
}

/* =====================================================
   VERDICT ENGINE
===================================================== */

function buildVerdict({
  calibratedScore,
  evidenceQuality,
  modelGap,
  segmentDeviation,
  segmentRange,
  activeModels
}) {
  const severeConflict =
    modelGap >= 35;

  const severeInstability =
    segmentDeviation >= 35 ||
    segmentRange >= 80;

  const limitedEvidence =
    activeModels < 2;

  if (
    severeConflict ||
    severeInstability ||
    limitedEvidence ||
    evidenceQuality < 0.32
  ) {
    return 'INCONCLUSIVE';
  }

  if (
    calibratedScore >= 82
  ) {
    return 'Strong AI evidence';
  }

  if (
    calibratedScore >= 65
  ) {
    return 'Likely AI';
  }

  if (
    calibratedScore <= 18
  ) {
    return 'Strong human evidence';
  }

  if (
    calibratedScore <= 35
  ) {
    return 'Likely human';
  }

  return 'Mixed / uncertain';
}

/* =====================================================
   CONFIDENCE
===================================================== */

function calculateConfidence(
  evidenceQuality,
  words
) {
  let confidence =
    evidenceQuality *
    100;

  if (
    words < 150
  ) {
    confidence -=
      8;
  }

  return Math.round(
    Math.max(
      10,
      Math.min(
        95,
        confidence
      )
    )
  );
}

/* =====================================================
   CORE ANALYSIS
===================================================== */

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
      'Paste at least 80 words for a meaningful analysis.'
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

  let tmrDocument =
    50;

  let modernDocument =
    50;

  let tmrWorked =
    true;

  let modernWorked =
    true;

  const tmrSegments =
    [];

  /*
    TMR
  */

  try {
    const tmr =
      await loadTMR();

    progress(
      18,
      'Running TMR…'
    );

    tmrDocument =
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
        25 +
        Math.round(
          (
            i /
            Math.max(
              chunks.length,
              1
            )
          ) *
          25
        ),
        `Analyzing segment ${i + 1}/${chunks.length}`
      );

      tmrSegments.push(
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
      'TMR error:',
      error
    );

    tmrWorked =
      false;

    tmrSegments.push(
      ...chunks.map(
        () => 50
      )
    );
  }

  /*
    ModernBERT
  */

  try {
    const modern =
      await loadModern();

    progress(
      68,
      'Running ModernBERT…'
    );

    modernDocument =
      await classify(
        modern,
        value
      );
  } catch (
    error
  ) {
    console.error(
      'ModernBERT error:',
      error
    );

    modernWorked =
      false;
  }

  progress(
    86,
    'Calibrating evidence…'
  );

  const activeModels =
    Number(tmrWorked) +
    Number(modernWorked);

  const modelGap =
    (
      tmrWorked &&
      modernWorked
    )
      ? Math.abs(
          tmrDocument -
          modernDocument
        )
      : 0;

  const segmentDeviation =
    standardDeviation(
      tmrSegments
    );

  const segmentRange =
    tmrSegments.length
      ? (
          Math.max(
            ...tmrSegments
          ) -
          Math.min(
            ...tmrSegments
          )
        )
      : 0;

  const rawSignal =
    calculateRawSignal(
      tmrDocument,
      modernDocument,
      tmrWorked,
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
      activeModels
    });

  const calibratedScore =
    calibrateScore(
      rawSignal,
      evidenceQuality
    );

  const confidence =
    calculateConfidence(
      evidenceQuality,
      profile.words
    );

  const verdict =
    buildVerdict({
      calibratedScore,
      evidenceQuality,
      modelGap,
      segmentDeviation,
      segmentRange,
      activeModels
    });

  const result = {
    rawSignal,

    calibratedScore,

    evidenceQuality,

    confidence,

    verdict,

    tmrDocument,

    modernDocument,

    tmrWorked,

    modernWorked,

    activeModels,

    modelGap:
      Math.round(
        modelGap
      ),

    segmentDeviation:
      Math.round(
        segmentDeviation
      ),

    segmentRange:
      Math.round(
        segmentRange
      ),

    profile,

    language,

    chunks,

    segmentScores:
      tmrSegments
  };

  renderV42(
    result
  );

  saveHistory(
    result
  );

  progress(
    100,
    'Calibration complete'
  );

  $('modelState')
    .textContent =
      activeModels === 2
        ? 'V4.2 calibrated engine ready ✓'
        : 'Limited evidence mode';

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
      askBenchmarkLabel(
        result
      );
    },
    600
  );
}

/* =====================================================
   RENDER
===================================================== */

function escapeHTML(
  value
) {
  return value.replace(
    /[&<>"']/g,
    character =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      }[character])
  );
}

function renderV42(
  result
) {
  $('report')
    .classList
    .remove('hidden');

  /*
    IMPORTANT:
    Until final redesign, the page still says
    AI INVOLVEMENT.

    The number shown now is the
    calibrated score, NOT raw model output.
  */

  $('score')
    .textContent =
      result.calibratedScore +
      '%';

  $('scaleFill')
    .style
    .width =
      result.calibratedScore +
      '%';

  $('verdict')
    .textContent =
      result.verdict;

  const confidenceLabel =
    result.confidence >= 70
      ? 'High'
      : result.confidence >= 45
        ? 'Medium'
        : 'Low';

  $('confidence')
    .textContent =
      `Confidence: ${confidenceLabel} (${result.confidence}%)`;

  $('explain')
    .textContent =
`Raw detector signal: ${result.rawSignal}%.
Calibrated score: ${result.calibratedScore}%.
Evidence quality: ${Math.round(result.evidenceQuality * 100)}%.

TMR: ${result.tmrDocument}%.
ModernBERT: ${result.modernDocument}%.
Model disagreement: ${result.modelGap} points.
Segment range: ${result.segmentRange} points.`;

  const human =
    100 -
    result.calibratedScore;

  const uncertainty =
    100 -
    result.confidence;

  $('humanVal')
    .textContent =
      `${human}%`;

  $('aiVal')
    .textContent =
      `${result.calibratedScore}%`;

  $('uncertainVal')
    .textContent =
      `${uncertainty}%`;

  $('humanBar')
    .style
    .width =
      `${human}%`;

  $('aiBar')
    .style
    .width =
      `${result.calibratedScore}%`;

  $('uncertainBar')
    .style
    .width =
      `${uncertainty}%`;

  $('engineBadge')
    .textContent =
      'V4.2 • CALIBRATED CONSENSUS';

  const evidence = [
    [
      'Raw detector signal',
      `${result.rawSignal}%`,
      'Raw'
    ],

    [
      'Calibrated score',
      `${result.calibratedScore}%`,
      'Adjusted'
    ],

    [
      'Evidence quality',
      `${Math.round(result.evidenceQuality * 100)}%`,
      result.evidenceQuality >= 0.7
        ? 'Strong'
        : result.evidenceQuality >= 0.4
          ? 'Medium'
          : 'Weak'
    ],

    [
      'TMR detector',
      result.tmrWorked
        ? `${result.tmrDocument}% AI signal`
        : 'Unavailable',
      'Model A'
    ],

    [
      'ModernBERT detector',
      result.modernWorked
        ? `${result.modernDocument}% AI signal`
        : 'Unavailable',
      'Model B'
    ],

    [
      'Model disagreement',
      `${result.modelGap} points`,
      result.modelGap >= 35
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment deviation',
      `${result.segmentDeviation}`,
      result.segmentDeviation >= 35
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Segment range',
      `${result.segmentRange} points`,
      result.segmentRange >= 80
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Language fit',
      result.language === 'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English detected — reliability reduced.',
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
      result.profile.words,

    Sentences:
      result.profile.sentences,

    'Avg. words / sentence':
      result.profile
        .avgSentenceLength
        .toFixed(1),

    'Lexical diversity':
      Math.round(
        result.profile
          .lexicalDiversity *
          100
      ) + '%',

    Language:
      result.language,

    'Models active':
      `${result.activeModels}/2`,

    'Raw signal':
      `${result.rawSignal}%`,

    'Calibrated score':
      `${result.calibratedScore}%`,

    'Evidence quality':
      `${Math.round(result.evidenceQuality * 100)}%`,

    'Model disagreement':
      `${result.modelGap} pts`,

    'Segment deviation':
      result.segmentDeviation,

    'Segment range':
      `${result.segmentRange} pts`
  };

  $('metrics')
    .innerHTML =
      Object.entries(
        metrics
      )
        .map(
          ([key, value]) => `
<div class="metric">
  <span>${key}</span>
  <b>${value}</b>
</div>`
        )
        .join('');

  $('segments')
    .innerHTML =
      result.chunks
        .map(
          (
            chunk,
            index
          ) => {
            const score =
              result.segmentScores[
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
    <i style="width:${score}%"></i>
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

</div>`;
          }
        )
        .join('');

  $('report')
    .scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
}

/* =====================================================
   HISTORY
===================================================== */

function saveHistory(
  result
) {
  try {
    const history =
      JSON.parse(
        localStorage.getItem(
          HISTORY_STORAGE
        ) || '[]'
      );

    history.push({
      timestamp:
        new Date()
          .toISOString(),

      version:
        APP_VERSION,

      words:
        result.profile.words,

      language:
        result.language,

      rawSignal:
        result.rawSignal,

      calibratedScore:
        result.calibratedScore,

      evidenceQuality:
        result.evidenceQuality,

      confidence:
        result.confidence,

      verdict:
        result.verdict,

      tmr:
        result.tmrDocument,

      modern:
        result.modernDocument,

      modelGap:
        result.modelGap,

      segmentDeviation:
        result.segmentDeviation,

      segmentRange:
        result.segmentRange
    });

    localStorage.setItem(
      HISTORY_STORAGE,
      JSON.stringify(
        history.slice(-100)
      )
    );
  } catch (
    error
  ) {
    console.warn(
      'History save failed:',
      error
    );
  }
}

/* =====================================================
   BENCHMARK
===================================================== */

function loadBenchmark() {
  try {
    return JSON.parse(
      localStorage.getItem(
        BENCHMARK_STORAGE
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
    BENCHMARK_STORAGE,
    JSON.stringify(records)
  );
}

function askBenchmarkLabel(
  result
) {
  const answer =
    prompt(
`AI TRACE BENCHMARK

If you KNOW the true origin:

AI = definitely AI-generated
HUMAN = definitely human-written

Leave empty / Cancel to skip.`
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

  records.push({
    timestamp:
      new Date()
        .toISOString(),

    version:
      APP_VERSION,

    groundTruth:
      truth,

    words:
      result.profile.words,

    rawSignal:
      result.rawSignal,

    calibratedScore:
      result.calibratedScore,

    evidenceQuality:
      result.evidenceQuality,

    verdict:
      result.verdict,

    confidence:
      result.confidence,

    tmr:
      result.tmrDocument,

    modern:
      result.modernDocument,

    modelGap:
      result.modelGap,

    segmentDeviation:
      result.segmentDeviation,

    segmentRange:
      result.segmentRange
  });

  saveBenchmark(
    records
  );

  alert(
    `Benchmark saved.\nTotal samples: ${records.length}`
  );
}

/* =====================================================
   DEV ACCESS
===================================================== */

window.AITraceV42 = {
  benchmark() {
    return loadBenchmark();
  },

  history() {
    try {
      return JSON.parse(
        localStorage.getItem(
          HISTORY_STORAGE
        ) || '[]'
      );
    } catch {
      return [];
    }
  },

  clearBenchmark() {
    localStorage.removeItem(
      BENCHMARK_STORAGE
    );
  },

  clearHistory() {
    localStorage.removeItem(
      HISTORY_STORAGE
    );
  }
};
