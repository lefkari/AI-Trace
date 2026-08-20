/*
  ============================================================
  AI TRACE V6.4 — BATCH CALIBRATION WORKER
  PART 1 / 3

  Core:
  - TMR detector
  - E5-small detector
  - Conditional ModernBERT
  - Human counter-evidence
  - Domain estimation
  - Segment analysis
  - Adaptive reliability
  - Leave-one-out calibration
  - Batch queue infrastructure
  - Resume-safe local storage
  ============================================================
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;


/* ============================================================
   VERSION
============================================================ */

const VERSION = '6.4';


/* ============================================================
   MODELS
============================================================ */

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';


/* ============================================================
   STORAGE KEYS
============================================================ */

const BENCH_KEY =
  'aiTraceBenchmarkV64';

const HISTORY_KEY =
  'aiTraceHistoryV64';

const IMPORT_KEY =
  'aiTraceCalibrationQueueV64';

const WORKER_STATE_KEY =
  'aiTraceCalibrationWorkerStateV64';


/* ============================================================
   LEGACY BENCHMARK KEYS
============================================================ */

const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV63',
  'aiTraceBenchmarkV62',
  'aiTraceBenchmarkV61',
  'aiTraceBenchmarkV6',
  'aiTraceBenchmarkV54',
  'aiTraceBenchmarkV53',
  'aiTraceBenchmarkV52',
  'aiTraceBenchmarkV51'
];


/* ============================================================
   MODEL CACHE
============================================================ */

let tmrModel = null;
let e5Model = null;
let modernModel = null;


/* ============================================================
   WORKER STATE
============================================================ */

let workerRunning = false;
let workerPaused = false;
let workerAbortRequested = false;


/* ============================================================
   DOM
============================================================ */

const $ = id =>
  document.getElementById(id);

const textEl =
  $('text');


/* ============================================================
   GENERIC HELPERS
============================================================ */

function clamp(
  value,
  min = 0,
  max = 100
) {
  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );
}


function nowISO() {
  return new Date()
    .toISOString();
}


function sleep(ms) {
  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );
}


function wordCount(value) {
  const clean =
    String(
      value || ''
    )
      .trim();

  if (!clean) {
    return 0;
  }

  return clean
    .split(/\s+/)
    .filter(Boolean)
    .length;
}


function average(values) {
  const usable =
    values.filter(
      Number.isFinite
    );

  if (!usable.length) {
    return 0;
  }

  return usable.reduce(
    (sum, value) =>
      sum + value,
    0
  ) / usable.length;
}


function median(values) {
  const usable =
    values
      .filter(
        Number.isFinite
      )
      .slice()
      .sort(
        (a, b) =>
          a - b
      );

  if (!usable.length) {
    return 50;
  }

  const middle =
    Math.floor(
      usable.length / 2
    );

  if (
    usable.length % 2
  ) {
    return usable[
      middle
    ];
  }

  return (
    usable[middle - 1] +
    usable[middle]
  ) / 2;
}


function standardDeviation(values) {
  const usable =
    values.filter(
      Number.isFinite
    );

  if (!usable.length) {
    return 0;
  }

  const mean =
    average(
      usable
    );

  return Math.sqrt(
    average(
      usable.map(
        value =>
          (
            value -
            mean
          ) ** 2
      )
    )
  );
}


function percentage(
  numerator,
  denominator
) {
  if (!denominator) {
    return 0;
  }

  return Math.round(
    numerator /
    denominator *
    100
  );
}


function escapeHTML(value) {
  return String(
    value ?? ''
  ).replace(
    /[&<>"']/g,
    character =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      })[
        character
      ]
  );
}


function countMatches(
  value,
  regex
) {
  return (
    String(value)
      .match(regex) ||
    []
  ).length;
}


function detectLanguage(value) {
  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) || []
    ).length;

  const letters =
    (
      value.match(
        /\p{L}/gu
      ) || []
    ).length;

  if (!letters) {
    return 'Unknown';
  }

  return (
    latin / letters >= 0.82
  )
    ? 'English'
    : 'Non-English';
}


function isMobileDevice() {
  return (
    window.matchMedia(
      '(max-width: 768px)'
    ).matches ||
    /Android|iPhone|iPad|iPod/i.test(
      navigator.userAgent
    )
  );
}


/* ============================================================
   STORAGE
============================================================ */

function loadJSON(
  key,
  fallback = []
) {
  try {
    const raw =
      localStorage.getItem(
        key
      );

    return raw
      ? JSON.parse(raw)
      : fallback;

  } catch (error) {
    console.warn(
      `Storage read failed: ${key}`,
      error
    );

    return fallback;
  }
}


function saveJSON(
  key,
  value
) {
  try {
    localStorage.setItem(
      key,
      JSON.stringify(
        value
      )
    );

    return true;

  } catch (error) {
    console.warn(
      `Storage write failed: ${key}`,
      error
    );

    return false;
  }
}


/* ============================================================
   UI HELPERS
============================================================ */

function updateCount() {
  if (
    !$('count')
  ) {
    return;
  }

  $('count').textContent =
    `${wordCount(
      textEl?.value || ''
    )} words`;
}


function setProgress(
  percent,
  label
) {
  $('progress')
    ?.classList
    .remove(
      'hidden'
    );

  if ($('bar')) {
    $('bar').style.width =
      `${clamp(percent)}%`;
  }

  if ($('progressText')) {
    $('progressText').textContent =
      label;
  }
}


function hideProgress() {
  setTimeout(
    () => {
      $('progress')
        ?.classList
        .add(
          'hidden'
        );
    },
    450
  );
}


function setState(label) {
  if ($('modelState')) {
    $('modelState').textContent =
      label;
  }
}


function setWorkerUI(
  state,
  text,
  percent = null
) {
  if ($('batchWorkerState')) {
    $('batchWorkerState').textContent =
      state;
  }

  if ($('batchWorkerText')) {
    $('batchWorkerText').textContent =
      text;
  }

  if (
    percent !== null &&
    $('batchWorkerBar')
  ) {
    $('batchWorkerBar').style.width =
      `${clamp(percent)}%`;
  }
}


/* ============================================================
   DEMO
============================================================ */

function loadDemo() {
  if (!textEl) {
    return;
  }

  textEl.value =
`Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

One major advantage of artificial intelligence is its ability to process information at a scale that would be difficult for humans to match. Organizations can use automated systems to identify patterns in large datasets, detect anomalies, and improve operational efficiency.

However, artificial intelligence also introduces new challenges. Machine-generated content may contain factual errors, misleading statements, or fabricated information. As generated media becomes more realistic, determining the origin of digital content becomes increasingly difficult.

Reliable AI detection therefore requires careful evaluation, transparent limitations, multiple independent signals, and conservative handling of uncertain evidence.`;

  updateCount();
}


/* ============================================================
   DOCUMENT PROFILE
============================================================ */

function createProfile(value) {
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

  const paragraphs =
    value
      .split(/\n\s*\n/)
      .map(
        paragraph =>
          paragraph.trim()
      )
      .filter(Boolean);

  const lines =
    value
      .split(/\n/)
      .map(
        line =>
          line.trim()
      )
      .filter(Boolean);

  const cleanedWords =
    words
      .map(
        word =>
          word
            .toLowerCase()
            .replace(
              /[^\p{L}\p{N}']/gu,
              ''
            )
      )
      .filter(Boolean);

  const sentenceLengths =
    sentences.map(
      sentence =>
        wordCount(
          sentence
        )
    );

  const paragraphLengths =
    paragraphs.map(
      paragraph =>
        wordCount(
          paragraph
        )
    );

  const averageSentenceLength =
    average(
      sentenceLengths
    );

  const sentenceDeviation =
    standardDeviation(
      sentenceLengths
    );

  const punctuationTypes =
    [
      /,/g,
      /;/g,
      /:/g,
      /[—–-]/g,
      /["“”‘’']/g,
      /[()]/g
    ]
      .filter(
        regex =>
          (
            value.match(
              regex
            ) || []
          ).length > 0
      )
      .length;

  const quoteCount =
    countMatches(
      value,
      /["“”‘’]/g
    );

  const firstPerson =
    countMatches(
      value,
      /\b(I|me|my|mine|we|us|our|ours)\b/gi
    );

  const contractions =
    countMatches(
      value,
      /\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi
    );

  const transitions =
    countMatches(
      value,
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|additionally|nevertheless|in conclusion|as a result)\b/gi
    );

  const dialogueLines =
    lines.filter(
      line =>
        /^[“"'—-]/.test(
          line
        ) ||
        /[”"']$/.test(
          line
        )
    ).length;

  return {
    words:
      words.length,

    sentences:
      sentences.length,

    paragraphs:
      paragraphs.length,

    lines:
      lines.length,

    averageSentenceLength,

    sentenceDeviation,

    sentenceBurstiness:
      averageSentenceLength
        ? sentenceDeviation /
          averageSentenceLength
        : 0,

    paragraphDeviation:
      standardDeviation(
        paragraphLengths
      ),

    lexicalDiversity:
      new Set(
        cleanedWords
      ).size /
      Math.max(
        1,
        cleanedWords.length
      ),

    punctuationTypes,

    quoteCount,

    firstPerson,

    contractions,

    transitions,

    dialogueLines,

    averageLineLength:
      lines.length
        ? value.length /
          lines.length
        : value.length
  };
}


/* ============================================================
   DOMAIN ESTIMATION
============================================================ */

function estimateDomain(
  value,
  profile
) {
  const content =
    value.toLowerCase();

  const signals = {
    books:
      profile.quoteCount >= 5 ||
      profile.dialogueLines >= 2
        ? 5
        : 0,

    poetry:
      profile.lines >= 7 &&
      profile.averageLineLength < 65
        ? 5
        : 0,

    academic:
      countMatches(
        content,
        /\b(method|methodology|results|participants|dataset|experiment|analysis|hypothesis|significant|research|abstract|conclusion)\b/g
      ),

    news:
      countMatches(
        content,
        /\b(reuters|reported|officials|government|minister|president|announced|agency|according to)\b/g
      ),

    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price)\b/g
      ),

    social:
      countMatches(
        content,
        /\b(imo|lol|reddit|subreddit|tldr|edit:|upvote|downvote|thread)\b/g
      ),

    recipe:
      countMatches(
        content,
        /\b(cup|tablespoon|teaspoon|ingredients|oven|bake|stir|chop|minutes|serve)\b/g
      )
  };

  const sorted =
    Object.entries(
      signals
    )
      .sort(
        (a, b) =>
          b[1] - a[1]
      );

  const [
    domain,
    score
  ] =
    sorted[0] || [
      'general',
      0
    ];

  if (
    score < 2
  ) {
    return {
      domain:
        'general',

      confidence:
        'low'
    };
  }

  return {
    domain,

    confidence:
      score >= 5
        ? 'high'
        : score >= 3
          ? 'medium'
          : 'low'
  };
}


/* ============================================================
   HUMAN COUNTER-EVIDENCE
============================================================ */

function humanEvidence(
  profile,
  domain
) {
  let score = 0;

  const reasons = [];

  if (
    profile.sentenceBurstiness >= 0.70
  ) {
    score += 20;

    reasons.push(
      'High sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >= 0.45
  ) {
    score += 12;

    reasons.push(
      'Moderate sentence-length variation'
    );
  }

  if (
    profile.punctuationTypes >= 5
  ) {
    score += 12;

    reasons.push(
      'Rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >= 3
  ) {
    score += 6;
  }

  if (
    profile.firstPerson >= 4
  ) {
    score += 10;

    reasons.push(
      'Personal or subjective voice'
    );

  } else if (
    profile.firstPerson > 0
  ) {
    score += 5;
  }

  if (
    profile.contractions >= 3
  ) {
    score += 8;

    reasons.push(
      'Natural contraction usage'
    );
  }

  if (
    profile.quoteCount >= 6 ||
    profile.dialogueLines >= 2
  ) {
    score += 14;

    reasons.push(
      'Dialogue or quotation structure'
    );
  }

  if (
    profile.paragraphDeviation >= 18 &&
    profile.paragraphs >= 3
  ) {
    score += 8;

    reasons.push(
      'Irregular paragraph rhythm'
    );
  }

  if (
    profile.transitions >= 4
  ) {
    score -= 6;
  }

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {
    score += 10;

    reasons.push(
      'Literary-domain protection'
    );
  }

  return {
    score:
      clamp(
        Math.round(
          score
        )
      ),

    reasons
  };
}


/* ============================================================
   CHUNKING
============================================================ */

function chunkText(
  value,
  maxChars = 1300
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


/* ============================================================
   MODEL LOADERS
============================================================ */

async function loadTMR() {
  if (
    tmrModel
  ) {
    return tmrModel;
  }

  setState(
    'Loading TMR…'
  );

  setProgress(
    8,
    'Loading detector A…'
  );

  tmrModel =
    await pipeline(
      'text-classification',
      MODEL_TMR,
      {
        dtype:
          'q4f16'
      }
    );

  return tmrModel;
}


async function loadE5() {
  if (
    e5Model
  ) {
    return e5Model;
  }

  setState(
    'Loading E5-small…'
  );

  setProgress(
    18,
    'Loading detector B…'
  );

  e5Model =
    await pipeline(
      'text-classification',
      MODEL_E5,
      {
        dtype:
          'q4f16'
      }
    );

  return e5Model;
}


async function loadModern() {
  if (
    modernModel
  ) {
    return modernModel;
  }

  setState(
    'Loading ModernBERT…'
  );

  setProgress(
    70,
    'Loading detector C…'
  );

  modernModel =
    await pipeline(
      'text-classification',
      MODEL_MODERN,
      {
        dtype:
          'q4f16'
      }
    );

  return modernModel;
}


/* ============================================================
   MODEL OUTPUT NORMALIZATION
============================================================ */

function aiProbability(
  output
) {
  const results =
    (
      Array.isArray(
        output
      )
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
        item?.label ||
        ''
      )
        .toLowerCase();

    const score =
      Number(
        item?.score
      );

    if (
      !Number.isFinite(
        score
      )
    ) {
      continue;
    }

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
  model,
  value
) {
  const output =
    await model(
      value,
      {
        top_k:
          null,

        truncation:
          true
      }
    );

  return Math.round(
    aiProbability(
      output
    ) * 100
  );
}


/* ============================================================
   BENCHMARK NORMALIZATION
============================================================ */

function normalizeBenchmarkRecord(
  record
) {
  if (!record) {
    return null;
  }

  const truth =
    String(
      record.truth ||
      record.groundTruth ||
      ''
    )
      .trim()
      .toUpperCase();

  return {
    ...record,

    truth,

    domain:
      record.domain ||
      'general',

    version:
      record.version ||
      record.appVersion ||
      'legacy'
  };
}


/* ============================================================
   BENCHMARK MIGRATION
============================================================ */

function loadBench() {
  const current =
    loadJSON(
      BENCH_KEY,
      []
    );

  if (
    Array.isArray(current) &&
    current.length
  ) {
    return current
      .map(
        normalizeBenchmarkRecord
      )
      .filter(Boolean);
  }

  for (
    const key
    of LEGACY_BENCH_KEYS
  ) {
    const legacy =
      loadJSON(
        key,
        []
      );

    if (
      Array.isArray(legacy) &&
      legacy.length
    ) {
      const normalized =
        legacy
          .map(
            normalizeBenchmarkRecord
          )
          .filter(Boolean);

      saveJSON(
        BENCH_KEY,
        normalized
      );

      return normalized;
    }
  }

  return [];
}


function saveBench(records) {
  return saveJSON(
    BENCH_KEY,
    records
  );
}


/* ============================================================
   BINARY BENCHMARK RECORDS
============================================================ */

function binaryRecords(
  records = loadBench()
) {
  return records.filter(
    record =>
      record.truth === 'AI' ||
      record.truth === 'HUMAN'
  );
}


/* ============================================================
   BENCHMARK PREDICTION
============================================================ */

function benchmarkPrediction(
  record
) {
  const verdict =
    record?.consensus
      ?.verdict ||
    '';

  if (
    verdict ===
      'Strong AI evidence' ||
    verdict ===
      'Likely AI'
  ) {
    return 'AI';
  }

  if (
    verdict ===
      'Strong human evidence' ||
    verdict ===
      'Likely human'
  ) {
    return 'HUMAN';
  }

  return 'ABSTAIN';
}


/* ============================================================
   GENERIC METRICS
============================================================ */

function evaluatePredictions(
  rows,
  getPrediction
) {
  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;

  let aiAbstain = 0;
  let humanAbstain = 0;

  for (
    const row
    of rows
  ) {
    const predicted =
      getPrediction(
        row
      );

    if (
      predicted ===
      'ABSTAIN'
    ) {
      if (
        row.truth === 'AI'
      ) {
        aiAbstain++;

      } else if (
        row.truth === 'HUMAN'
      ) {
        humanAbstain++;
      }

      continue;
    }

    if (
      row.truth === 'AI' &&
      predicted === 'AI'
    ) {
      TP++;
    }

    if (
      row.truth === 'HUMAN' &&
      predicted === 'HUMAN'
    ) {
      TN++;
    }

    if (
      row.truth === 'HUMAN' &&
      predicted === 'AI'
    ) {
      FP++;
    }

    if (
      row.truth === 'AI' &&
      predicted === 'HUMAN'
    ) {
      FN++;
    }
  }

  const totalAI =
    rows.filter(
      row =>
        row.truth === 'AI'
    ).length;

  const totalHuman =
    rows.filter(
      row =>
        row.truth === 'HUMAN'
    ).length;

  const decided =
    TP +
    TN +
    FP +
    FN;

  return {
    total:
      rows.length,

    totalAI,

    totalHuman,

    TP,
    TN,
    FP,
    FN,

    decided,

    aiAbstain,
    humanAbstain,

    coverage:
      percentage(
        decided,
        rows.length
      ),

    selectiveAccuracy:
      percentage(
        TP + TN,
        decided
      ),

    precision:
      percentage(
        TP,
        TP + FP
      ),

    recall:
      percentage(
        TP,
        totalAI
      ),

    specificity:
      percentage(
        TN,
        totalHuman
      ),

    fpr:
      percentage(
        FP,
        totalHuman
      ),

    fnr:
      percentage(
        FN,
        totalAI
      ),

    aiAbstainRate:
      percentage(
        aiAbstain,
        totalAI
      ),

    humanAbstainRate:
      percentage(
        humanAbstain,
        totalHuman
      )
  };
}


/* ============================================================
   DETECTOR PREDICTION
============================================================ */

function detectorPrediction(
  record,
  detector
) {
  const score =
    Number(
      record.scores?.[
        detector
      ]
    );

  if (
    !Number.isFinite(score)
  ) {
    return 'ABSTAIN';
  }

  if (
    score >= 70
  ) {
    return 'AI';
  }

  if (
    score <= 30
  ) {
    return 'HUMAN';
  }

  return 'ABSTAIN';
}
/* ============================================================
   LEAVE-ONE-OUT SUPPORT
============================================================ */

function leaveOneOutRecords(
  records,
  excludeId
) {
  if (!excludeId) {
    return records;
  }

  return records.filter(
    record =>
      record.id !== excludeId
  );
}


/* ============================================================
   DETECTOR RELIABILITY METRICS
============================================================ */

function detectorReliabilityMetrics(
  detector,
  domain,
  records,
  excludeId = null
) {
  const clean =
    leaveOneOutRecords(
      binaryRecords(
        records
      ),
      excludeId
    );

  const globalRows =
    clean.filter(
      record =>
        Number.isFinite(
          Number(
            record.scores?.[
              detector
            ]
          )
        )
    );

  const domainRows =
    globalRows.filter(
      record =>
        (
          record.domain ||
          'general'
        ) === domain
    );

  const globalMetrics =
    evaluatePredictions(
      globalRows,
      record =>
        detectorPrediction(
          record,
          detector
        )
    );

  const domainMetrics =
    evaluatePredictions(
      domainRows,
      record =>
        detectorPrediction(
          record,
          detector
        )
    );

  return {
    globalRows,
    domainRows,
    globalMetrics,
    domainMetrics
  };
}


/* ============================================================
   RELIABILITY WEIGHT
============================================================ */

function calculateReliabilityWeight(
  metrics,
  sampleCount
) {
  /*
    Small datasets remain close to neutral.
  */

  if (
    sampleCount < 20
  ) {
    return 1;
  }

  const accuracy =
    metrics.selectiveAccuracy /
    100;

  const coverage =
    metrics.coverage /
    100;

  const fprSafety =
    1 -
    metrics.fpr /
    100;

  const fnrSafety =
    1 -
    metrics.fnr /
    100;

  let score =
    accuracy * 0.42 +
    coverage * 0.14 +
    fprSafety * 0.29 +
    fnrSafety * 0.15;

  /*
    Safety penalty:
    false positives matter a lot for AI detection.
  */

  if (
    metrics.fpr >= 40
  ) {
    score *= 0.75;
  }

  if (
    metrics.fpr >= 70
  ) {
    score *= 0.60;
  }

  /*
    Reliability only gradually gains influence.
  */

  const maturity =
    clamp(
      sampleCount / 120,
      0,
      1
    );

  const softened =
    1 +
    (
      score -
      1
    ) * maturity;

  return clamp(
    softened,
    0.30,
    1.30
  );
}


/* ============================================================
   DIRECTIONAL RELIABILITY
============================================================ */

function directionalReliability(
  detector,
  direction,
  domain,
  records,
  excludeId = null
) {
  const stats =
    detectorReliabilityMetrics(
      detector,
      domain,
      records,
      excludeId
    );

  const globalRows =
    stats.globalRows;

  const domainRows =
    stats.domainRows;

  function evaluateDirection(rows) {
    let predictions = 0;
    let correct = 0;

    for (
      const record
      of rows
    ) {
      const predicted =
        detectorPrediction(
          record,
          detector
        );

      if (
        predicted !== direction
      ) {
        continue;
      }

      predictions++;

      if (
        record.truth === direction
      ) {
        correct++;
      }
    }

    return {
      predictions,
      correct,

      accuracy:
        percentage(
          correct,
          predictions
        )
    };
  }

  const global =
    evaluateDirection(
      globalRows
    );

  const domainSpecific =
    evaluateDirection(
      domainRows
    );

  const globalReady =
    global.predictions >= 10;

  const domainReady =
    domainSpecific.predictions >= 6;

  let weight = 1;

  if (
    globalReady
  ) {
    weight =
      clamp(
        global.accuracy /
        100,
        0.25,
        1
      );
  }

  if (
    globalReady &&
    domainReady
  ) {
    weight =
      weight * 0.45 +
      clamp(
        domainSpecific.accuracy /
        100,
        0.20,
        1
      ) * 0.55;
  }

  return {
    ready:
      globalReady ||
      domainReady,

    weight:
      Number(
        clamp(
          weight,
          0.20,
          1.05
        ).toFixed(
          3
        )
      ),

    global,

    domain:
      domainSpecific
  };
}


/* ============================================================
   MODEL RELIABILITY PROFILE
============================================================ */

function buildModelReliability(
  domain,
  records = loadBench(),
  excludeId = null
) {
  const result = {};

  for (
    const detector
    of [
      'tmr',
      'e5',
      'modern'
    ]
  ) {
    const stats =
      detectorReliabilityMetrics(
        detector,
        domain,
        records,
        excludeId
      );

    const base =
      calculateReliabilityWeight(
        stats.globalMetrics,
        stats.globalRows.length
      );

    result[
      detector
    ] = {
      base,

      global:
        {
          samples:
            stats.globalRows.length,

          metrics:
            stats.globalMetrics
        },

      domain:
        {
          samples:
            stats.domainRows.length,

          metrics:
            stats.domainMetrics
        },

      ai:
        directionalReliability(
          detector,
          'AI',
          domain,
          records,
          excludeId
        ),

      human:
        directionalReliability(
          detector,
          'HUMAN',
          domain,
          records,
          excludeId
        )
    };
  }

  return result;
}


/* ============================================================
   BENCHMARK READINESS
============================================================ */

function benchmarkReadiness(
  records = loadBench()
) {
  const rows =
    binaryRecords(
      records
    );

  const ai =
    rows.filter(
      record =>
        record.truth === 'AI'
    ).length;

  const human =
    rows.filter(
      record =>
        record.truth === 'HUMAN'
    ).length;

  const domains =
    new Set(
      rows.map(
        record =>
          record.domain ||
          'general'
      )
    ).size;

  let level =
    'COLLECTING';

  if (
    rows.length >= 250 &&
    ai >= 100 &&
    human >= 100 &&
    domains >= 5
  ) {
    level =
      'STRONG';

  } else if (
    rows.length >= 120 &&
    ai >= 50 &&
    human >= 50 &&
    domains >= 4
  ) {
    level =
      'GOOD';

  } else if (
    rows.length >= 50 &&
    ai >= 20 &&
    human >= 20 &&
    domains >= 3
  ) {
    level =
      'EARLY';

  } else if (
    rows.length >= 20 &&
    ai >= 8 &&
    human >= 8
  ) {
    level =
      'EXPERIMENTAL';
  }

  return {
    level,
    total:
      rows.length,
    ai,
    human,
    domains
  };
}


/* ============================================================
   OUTLIER DETECTION
============================================================ */

function detectModelOutlier(
  scores
) {
  const entries =
    Object.entries(
      scores
    )
      .filter(
        (
          [
            ,
            value
          ]
        ) =>
          Number.isFinite(
            value
          )
      );

  if (
    entries.length < 3
  ) {
    return {
      detected:
        false,

      detector:
        null,

      distance:
        0
    };
  }

  const values =
    entries.map(
      (
        [
          ,
          value
        ]
      ) =>
        value
    );

  const reference =
    median(
      values
    );

  const distances =
    entries.map(
      (
        [
          detector,
          score
        ]
      ) => ({
        detector,
        score,

        distance:
          Math.abs(
            score -
            reference
          )
      })
    )
      .sort(
        (
          a,
          b
        ) =>
          b.distance -
          a.distance
      );

  const first =
    distances[0];

  const second =
    distances[1];

  const detected =
    first.distance >= 30 &&
    (
      first.distance -
      second.distance
    ) >= 14;

  return {
    detected,

    detector:
      detected
        ? first.detector
        : null,

    distance:
      Math.round(
        first.distance
      ),

    reference:
      Math.round(
        reference
      )
  };
}


/* ============================================================
   SEGMENT ANALYSIS
============================================================ */

function analyzeSegments(
  segmentScores
) {
  const valid =
    segmentScores.filter(
      Number.isFinite
    );

  if (
    !valid.length
  ) {
    return {
      mean:
        50,

      deviation:
        50,

      range:
        100,

      stability:
        0,

      aiSegments:
        0,

      humanSegments:
        0,

      uncertainSegments:
        0,

      mixed:
        true
    };
  }

  const mean =
    average(
      valid
    );

  const deviation =
    standardDeviation(
      valid
    );

  const range =
    Math.max(
      ...valid
    ) -
    Math.min(
      ...valid
    );

  const aiSegments =
    valid.filter(
      score =>
        score >= 70
    ).length;

  const humanSegments =
    valid.filter(
      score =>
        score <= 30
    ).length;

  const uncertainSegments =
    valid.length -
    aiSegments -
    humanSegments;

  let stability =
    100;

  stability -=
    Math.min(
      45,
      deviation * 1.25
    );

  stability -=
    Math.min(
      35,
      range * 0.35
    );

  if (
    aiSegments > 0 &&
    humanSegments > 0
  ) {
    stability -= 10;
  }

  return {
    mean:
      Math.round(
        mean
      ),

    deviation:
      Math.round(
        deviation
      ),

    range:
      Math.round(
        range
      ),

    stability:
      clamp(
        Math.round(
          stability
        )
      ),

    aiSegments,

    humanSegments,

    uncertainSegments,

    mixed:
      aiSegments > 0 &&
      humanSegments > 0
  };
}


/* ============================================================
   MODEL AGREEMENT
============================================================ */

function calculateModelAgreement(
  scores
) {
  const values =
    Object.values(
      scores
    )
      .filter(
        Number.isFinite
      );

  if (
    values.length <= 1
  ) {
    return {
      active:
        values.length,

      agreement:
        0,

      spread:
        100,

      deviation:
        50
    };
  }

  const spread =
    Math.max(
      ...values
    ) -
    Math.min(
      ...values
    );

  const deviation =
    standardDeviation(
      values
    );

  const agreement =
    clamp(
      Math.round(
        100 -
        spread * 1.10 -
        deviation * 0.50
      )
    );

  return {
    active:
      values.length,

    agreement,

    spread:
      Math.round(
        spread
      ),

    deviation:
      Math.round(
        deviation
      )
  };
}


/* ============================================================
   EVIDENCE SUFFICIENCY
============================================================ */

function calculateEvidenceSufficiency({
  profile,
  language,
  domain,
  modelAgreement,
  segmentAnalysis,
  outlier,
  human,
  thirdUsed
}) {
  let score = 100;

  if (
    profile.words < 100
  ) {
    score -= 30;

  } else if (
    profile.words < 150
  ) {
    score -= 18;

  } else if (
    profile.words < 220
  ) {
    score -= 8;
  }

  if (
    language !== 'English'
  ) {
    score -= 35;
  }

  if (
    modelAgreement.active === 1
  ) {
    score -= 35;

  } else if (
    modelAgreement.active === 2
  ) {
    score -= 10;
  }

  score -=
    Math.round(
      (
        100 -
        modelAgreement.agreement
      ) * 0.35
    );

  score -=
    Math.round(
      (
        100 -
        segmentAnalysis.stability
      ) * 0.20
    );

  if (
    outlier.detected
  ) {
    score -= 5;
  }

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {
    score -= 10;
  }

  if (
    human.score >= 55
  ) {
    score -= 5;
  }

  if (
    thirdUsed &&
    modelAgreement.active >= 3
  ) {
    score += 5;
  }

  score =
    clamp(
      Math.round(
        score
      )
    );

  let level =
    'INSUFFICIENT';

  if (
    score >= 75
  ) {
    level =
      'STRONG';

  } else if (
    score >= 55
  ) {
    level =
      'MODERATE';

  } else if (
    score >= 40
  ) {
    level =
      'WEAK';
  }

  return {
    score,
    level
  };
}


/* ============================================================
   MODEL WEIGHT FOR CURRENT SCORE
============================================================ */

function modelWeightForCurrentScore(
  detector,
  score,
  reliability,
  readiness
) {
  const profile =
    reliability?.[
      detector
    ];

  if (!profile) {
    return 1;
  }

  const direction =
    score >= 50
      ? 'ai'
      : 'human';

  let weight =
    (
      profile.base * 0.68 +
      (
        profile[
          direction
        ]?.weight ??
        1
      ) * 0.32
    );

  /*
    With tiny datasets weights remain very close to 1.
  */

  if (
    readiness.level === 'COLLECTING'
  ) {
    weight =
      clamp(
        weight,
        0.90,
        1.08
      );
  }

  if (
    readiness.level === 'EXPERIMENTAL'
  ) {
    weight =
      clamp(
        weight,
        0.75,
        1.15
      );
  }

  return clamp(
    weight,
    0.25,
    1.35
  );
}


/* ============================================================
   ADAPTIVE MODEL SIGNAL
============================================================ */

function adaptiveModelSignal({
  scores,
  reliability,
  readiness,
  outlier
}) {
  const values = [];
  const weights = [];
  const details = {};

  for (
    const detector
    of [
      'tmr',
      'e5',
      'modern'
    ]
  ) {
    const score =
      scores[
        detector
      ];

    if (
      !Number.isFinite(
        score
      )
    ) {
      continue;
    }

    let weight =
      modelWeightForCurrentScore(
        detector,
        score,
        reliability,
        readiness
      );

    if (
      outlier.detected &&
      outlier.detector === detector
    ) {
      weight *= 0.45;
    }

    weight =
      clamp(
        weight,
        0.20,
        1.40
      );

    values.push(
      score
    );

    weights.push(
      weight
    );

    details[
      detector
    ] = {
      score,

      weight:
        Number(
          weight.toFixed(
            3
          )
        ),

      outlier:
        outlier.detected &&
        outlier.detector === detector
    };
  }

  let weightedTotal = 0;
  let weightTotal = 0;

  for (
    let i = 0;
    i < values.length;
    i++
  ) {
    weightedTotal +=
      values[i] *
      weights[i];

    weightTotal +=
      weights[i];
  }

  const weighted =
    weightTotal
      ? weightedTotal /
        weightTotal
      : 50;

  const robustMedian =
    median(
      values
    );

  const signal =
    Math.round(
      weighted * 0.62 +
      robustMedian * 0.38
    );

  return {
    signal:
      clamp(
        signal
      ),

    weighted:
      Math.round(
        weighted
      ),

    median:
      Math.round(
        robustMedian
      ),

    details
  };
}


/* ============================================================
   THIRD MODEL ROUTING
============================================================ */

function shouldUseThirdModel({
  scores,
  human,
  segmentScores,
  domain,
  words,
  language
}) {
  if (
    isMobileDevice()
  ) {
    return false;
  }

  if (
    language !== 'English'
  ) {
    return true;
  }

  if (
    !Number.isFinite(
      scores.tmr
    ) ||
    !Number.isFinite(
      scores.e5
    )
  ) {
    return true;
  }

  const gap =
    Math.abs(
      scores.tmr -
      scores.e5
    );

  const quickMedian =
    median([
      scores.tmr,
      scores.e5
    ]);

  const segment =
    analyzeSegments(
      segmentScores
    );

  return (
    words < 180 ||
    gap >= 18 ||
    (
      quickMedian >= 30 &&
      quickMedian <= 82
    ) ||
    segment.range >= 45 ||
    segment.deviation >= 20 ||
    segment.mixed ||
    human.score >= 40 ||
    domain === 'books' ||
    domain === 'poetry'
  );
}


/* ============================================================
   CONSENSUS ENGINE
============================================================ */

function buildConsensus({
  scores,
  profile,
  segmentScores,
  language,
  domain,
  human,
  thirdUsed,
  benchmarkRecords,
  excludeBenchmarkId = null
}) {
  const readiness =
    benchmarkReadiness(
      benchmarkRecords
    );

  const reliability =
    buildModelReliability(
      domain,
      benchmarkRecords,
      excludeBenchmarkId
    );

  const outlier =
    detectModelOutlier(
      scores
    );

  const modelAgreement =
    calculateModelAgreement(
      scores
    );

  const segmentAnalysis =
    analyzeSegments(
      segmentScores
    );

  const adaptive =
    adaptiveModelSignal({
      scores,
      reliability,
      readiness,
      outlier
    });

  const raw =
    adaptive.signal;

  const sufficiency =
    calculateEvidenceSufficiency({
      profile,
      language,
      domain,
      modelAgreement,
      segmentAnalysis,
      outlier,
      human,
      thirdUsed
    });

  const disagreement =
    1 -
    modelAgreement.agreement /
    100;

  const humanPenalty =
    human.score *
    (
      0.08 +
      disagreement * 0.34
    ) *
    (
      raw /
      100
    );

  let calibrated =
    clamp(
      Math.round(
        raw -
        humanPenalty
      )
    );

  /*
    Literary false-positive protection.
  */

  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    human.score >= 40 &&
    modelAgreement.agreement < 55
  ) {
    calibrated =
      Math.min(
        calibrated,
        64
      );
  }

  /*
    Extreme score guard.
  */

  if (
    calibrated >= 95 &&
    (
      modelAgreement.spread > 18 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 80
    )
  ) {
    calibrated = 94;
  }

  if (
    calibrated <= 5 &&
    (
      modelAgreement.spread > 18 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 75
    )
  ) {
    calibrated = 6;
  }

  let verdict =
    'INCONCLUSIVE';

  const severeConflict =
    modelAgreement.spread >= 70;

  const highConflict =
    modelAgreement.spread >= 45 ||
    modelAgreement.agreement <= 30;

  if (
    language === 'English' &&
    modelAgreement.active >= 2 &&
    calibrated >= 86 &&
    sufficiency.score >= 74 &&
    modelAgreement.agreement >= 48 &&
    !severeConflict &&
    human.score < 50
  ) {
    verdict =
      'Strong AI evidence';

  } else if (
    language === 'English' &&
    modelAgreement.active >= 2 &&
    calibrated >= 74 &&
    sufficiency.score >= 62 &&
    modelAgreement.agreement >= 38 &&
    modelAgreement.spread < 55 &&
    human.score < 55
  ) {
    verdict =
      'Likely AI';

  } else if (
    language === 'English' &&
    modelAgreement.active >= 2 &&
    calibrated <= 18 &&
    human.score >= 50 &&
    sufficiency.score >= 60 &&
    modelAgreement.agreement >= 45
  ) {
    verdict =
      'Strong human evidence';

  } else if (
    language === 'English' &&
    modelAgreement.active >= 2 &&
    calibrated <= 34 &&
    human.score >= 40 &&
    sufficiency.score >= 50 &&
    modelAgreement.spread < 50
  ) {
    verdict =
      'Likely human';
  }

  if (
    highConflict &&
    verdict !== 'INCONCLUSIVE'
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    sufficiency.score < 55
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    language !== 'English'
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  const confidence =
    verdict === 'INCONCLUSIVE'
      ? Math.min(
          55,
          Math.round(
            sufficiency.score * 0.45 +
            modelAgreement.agreement * 0.35 +
            segmentAnalysis.stability * 0.20
          )
        )
      : clamp(
          Math.round(
            sufficiency.score * 0.45 +
            modelAgreement.agreement * 0.35 +
            segmentAnalysis.stability * 0.20
          )
        );

  const uncertainty =
    clamp(
      100 -
      confidence,
      5,
      95
    );

  return {
    raw,

    weightedRaw:
      adaptive.weighted,

    rawMedian:
      adaptive.median,

    calibrated,

    verdict,

    confidence,

    uncertainty,

    activeModels:
      modelAgreement.active,

    modelWeights:
      adaptive.details,

    modelAgreement,

    modelSpread:
      modelAgreement.spread,

    modelSD:
      modelAgreement.deviation,

    segmentAnalysis,

    segmentRange:
      segmentAnalysis.range,

    segmentSD:
      segmentAnalysis.deviation,

    outlier,

    sufficiency,

    reliability,

    readiness,

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    thirdUsed
  };
}


/* ============================================================
   SINGLE SAMPLE ANALYZER
============================================================ */

async function analyzeSample({
  value,
  truth = null,
  source = '',
  suppliedDomain = null,
  excludeBenchmarkId = null,
  forBatch = false
}) {
  const words =
    wordCount(
      value
    );

  if (
    words < 30
  ) {
    throw new Error(
      'Sample text is too short.'
    );
  }

  const language =
    detectLanguage(
      value
    );

  const profile =
    createProfile(
      value
    );

  const domainInfo =
    suppliedDomain &&
    suppliedDomain !== 'auto'
      ? {
          domain:
            suppliedDomain,

          confidence:
            'supplied'
        }
      : estimateDomain(
          value,
          profile
        );

  const human =
    humanEvidence(
      profile,
      domainInfo.domain
    );

  const chunks =
    chunkText(
      value
    );

  const scores = {
    tmr:
      NaN,

    e5:
      NaN,

    modern:
      NaN
  };

  let segmentScores =
    [];

  let thirdUsed =
    false;

  /*
    TMR
  */

  try {
    const modelA =
      await loadTMR();

    scores.tmr =
      await classify(
        modelA,
        value
      );

    for (
      const chunk
      of chunks
    ) {
      try {
        segmentScores.push(
          await classify(
            modelA,
            chunk
          )
        );

      } catch {
        segmentScores.push(
          50
        );
      }
    }

  } catch (error) {
    console.error(
      'TMR sample analysis failed:',
      error
    );
  }

  /*
    E5
  */

  try {
    const modelB =
      await loadE5();

    scores.e5 =
      await classify(
        modelB,
        value
      );

  } catch (error) {
    console.error(
      'E5 sample analysis failed:',
      error
    );
  }

  /*
    Third model routing.
  */

  if (
    !isMobileDevice() &&
    Number.isFinite(
      scores.tmr
    ) &&
    Number.isFinite(
      scores.e5
    )
  ) {
    thirdUsed =
      shouldUseThirdModel({
        scores,
        human,
        segmentScores,
        domain:
          domainInfo.domain,
        words,
        language
      });
  }

  if (
    thirdUsed
  ) {
    try {
      const modelC =
        await loadModern();

      scores.modern =
        await classify(
          modelC,
          value
        );

    } catch (error) {
      console.error(
        'ModernBERT sample analysis failed:',
        error
      );

      thirdUsed =
        false;
    }
  }

  if (
    !segmentScores.length
  ) {
    segmentScores =
      chunks.map(
        () => 50
      );
  }

  const benchmarkRecords =
    loadBench();

  const consensus =
    buildConsensus({
      scores,
      profile,
      segmentScores,
      language,
      domain:
        domainInfo.domain,
      human,
      thirdUsed,
      benchmarkRecords,
      excludeBenchmarkId
    });

  return {
    version:
      VERSION,

    timestamp:
      nowISO(),

    truth,

    source,

    text:
      forBatch
        ? value
        : undefined,

    words,

    language,

    domain:
      domainInfo.domain,

    domainConfidence:
      domainInfo.confidence,

    profile,

    human,

    scores,

    segmentScores,

    consensus
  };
}


/* ============================================================
   CALIBRATION QUEUE STORAGE
============================================================ */

function loadCalibrationQueue() {
  return loadJSON(
    IMPORT_KEY,
    []
  );
}


function saveCalibrationQueue(
  queue
) {
  return saveJSON(
    IMPORT_KEY,
    queue
  );
}


/* ============================================================
   WORKER STATE STORAGE
============================================================ */

function saveWorkerState(
  state
) {
  saveJSON(
    WORKER_STATE_KEY,
    state
  );
}


function loadWorkerState() {
  return loadJSON(
    WORKER_STATE_KEY,
    {
      processed:
        0,

      failed:
        0,

      lastUpdated:
        null
    }
  );
}


/* ============================================================
   QUEUE COUNTS
============================================================ */

function queueSummary(
  queue = loadCalibrationQueue()
) {
  const pending =
    queue.filter(
      item =>
        item.status === 'PENDING'
    ).length;

  const running =
    queue.filter(
      item =>
        item.status === 'RUNNING'
    ).length;

  const complete =
    queue.filter(
      item =>
        item.status === 'COMPLETE'
    ).length;

  const failed =
    queue.filter(
      item =>
        item.status === 'FAILED'
    ).length;

  return {
    total:
      queue.length,

    pending,

    running,

    complete,

    failed
  };
}


/* ============================================================
   RECOVER INTERRUPTED JOBS
============================================================ */

function recoverInterruptedQueue() {
  const queue =
    loadCalibrationQueue();

  let changed =
    false;

  for (
    const item
    of queue
  ) {
    if (
      item.status === 'RUNNING'
    ) {
      item.status =
        'PENDING';

      item.recoveredAt =
        nowISO();

      changed =
        true;
    }
  }

  if (
    changed
  ) {
    saveCalibrationQueue(
      queue
    );
  }
}


/* ============================================================
   UNIQUE BENCHMARK ID
============================================================ */

function nextBenchmarkId(
  truth,
  records
) {
  const prefix = {
    AI:
      'A',

    HUMAN:
      'H',

    MIXED:
      'M',

    UNKNOWN:
      'U'
  }[
    truth
  ] || 'X';

  const count =
    records.filter(
      record =>
        record.truth === truth
    ).length + 1;

  return `${prefix}-${String(
    count
  ).padStart(
    4,
    '0'
  )}`;
}


/* ============================================================
   SAVE ANALYZED BATCH RECORD
============================================================ */

function saveAnalyzedBatchRecord(
  queueItem,
  analysis
) {
  const records =
    loadBench();

  const id =
    nextBenchmarkId(
      queueItem.truth,
      records
    );

  const record = {
    id,

    truth:
      queueItem.truth,

    source:
      queueItem.source ||
      '',

    imported:
      true,

    importId:
      queueItem.importId,

    savedAt:
      nowISO(),

    predictionFrozen:
      true,

    ...analysis,

    text:
      queueItem.text
  };

  records.push(
    record
  );

  saveBench(
    records
  );

  return id;
}


/* ============================================================
   BATCH SAMPLE PROCESSOR
============================================================ */

async function processQueueItem(
  item
) {
  const analysis =
    await analyzeSample({
      value:
        item.text,

      truth:
        item.truth,

      source:
        item.source,

      suppliedDomain:
        item.domain ||
        null,

      forBatch:
        true
    });

  const benchmarkId =
    saveAnalyzedBatchRecord(
      item,
      analysis
    );

  return {
    benchmarkId,
    analysis
  };
}


/* ============================================================
   BATCH WORKER
============================================================ */

async function runCalibrationWorker() {
  if (
    workerRunning
  ) {
    return;
  }

  let queue =
    loadCalibrationQueue();

  const summary =
    queueSummary(
      queue
    );

  if (
    !summary.pending
  ) {
    setWorkerUI(
      'Idle',
      'No pending calibration samples.',
      summary.total
        ? 100
        : 0
    );

    return;
  }

  workerRunning =
    true;

  workerPaused =
    false;

  workerAbortRequested =
    false;

  if (
    $('runCalibrationQueue')
  ) {
    $('runCalibrationQueue').disabled =
      true;
  }

  setWorkerUI(
    'Running',
    `Preparing ${summary.pending} pending samples…`,
    summary.total
      ? summary.complete /
        summary.total *
        100
      : 0
  );

  try {
    while (true) {
      if (
        workerAbortRequested
      ) {
        break;
      }

      while (
        workerPaused
      ) {
        setWorkerUI(
          'Paused',
          'Calibration worker is paused.'
        );

        await sleep(
          400
        );

        if (
          workerAbortRequested
        ) {
          break;
        }
      }

      if (
        workerAbortRequested
      ) {
        break;
      }

      queue =
        loadCalibrationQueue();

      const nextIndex =
        queue.findIndex(
          item =>
            item.status ===
            'PENDING'
        );

      if (
        nextIndex === -1
      ) {
        break;
      }

      const item =
        queue[
          nextIndex
        ];

      item.status =
        'RUNNING';

      item.startedAt =
        nowISO();

      saveCalibrationQueue(
        queue
      );

      const before =
        queueSummary(
          queue
        );

      setWorkerUI(
        'Running',
        `Analyzing ${before.complete + 1}/${before.total} · ${item.truth} · ${wordCount(item.text)} words`,
        before.total
          ? before.complete /
            before.total *
            100
          : 0
      );

      try {
        const result =
          await processQueueItem(
            item
          );

        queue =
          loadCalibrationQueue();

        const currentIndex =
          queue.findIndex(
            row =>
              row.importId ===
              item.importId
          );

        if (
          currentIndex !== -1
        ) {
          queue[
            currentIndex
          ].status =
            'COMPLETE';

          queue[
            currentIndex
          ].completedAt =
            nowISO();

          queue[
            currentIndex
          ].benchmarkId =
            result.benchmarkId;

          queue[
            currentIndex
          ].error =
            null;
        }

        saveCalibrationQueue(
          queue
        );

        const workerState =
          loadWorkerState();

        workerState.processed =
          (
            workerState.processed ||
            0
          ) + 1;

        workerState.lastUpdated =
          nowISO();

        saveWorkerState(
          workerState
        );

      } catch (error) {
        console.error(
          'Batch sample failed:',
          error
        );

        queue =
          loadCalibrationQueue();

        const currentIndex =
          queue.findIndex(
            row =>
              row.importId ===
              item.importId
          );

        if (
          currentIndex !== -1
        ) {
          queue[
            currentIndex
          ].status =
            'FAILED';

          queue[
            currentIndex
          ].completedAt =
            nowISO();

          queue[
            currentIndex
          ].error =
            String(
              error?.message ||
              error
            );
        }

        saveCalibrationQueue(
          queue
        );

        const workerState =
          loadWorkerState();

        workerState.failed =
          (
            workerState.failed ||
            0
          ) + 1;

        workerState.lastUpdated =
          nowISO();

        saveWorkerState(
          workerState
        );
      }

      const after =
        queueSummary(
          loadCalibrationQueue()
        );

      const completed =
        after.complete +
        after.failed;

      setWorkerUI(
        'Running',
        `${completed}/${after.total} processed · ${after.failed} failed`,
        after.total
          ? completed /
            after.total *
            100
          : 0
      );

      /*
        Give browser UI / memory cleanup some time.
      */

      await sleep(
        350
      );

      renderDatasetManager?.();
      renderCalibrationLab?.();
    }

    const finalQueue =
      loadCalibrationQueue();

    const finalSummary =
      queueSummary(
        finalQueue
      );

    if (
      workerAbortRequested
    ) {
      setWorkerUI(
        'Stopped',
        `${finalSummary.complete}/${finalSummary.total} completed · ${finalSummary.pending} pending`,
        finalSummary.total
          ? finalSummary.complete /
            finalSummary.total *
            100
          : 0
      );

    } else if (
      workerPaused
    ) {
      setWorkerUI(
        'Paused',
        `${finalSummary.complete}/${finalSummary.total} completed · ${finalSummary.pending} pending`,
        finalSummary.total
          ? finalSummary.complete /
            finalSummary.total *
            100
          : 0
      );

    } else {
      setWorkerUI(
        'Complete',
        `${finalSummary.complete} completed · ${finalSummary.failed} failed`,
        100
      );
    }

  } finally {
    workerRunning =
      false;

    if (
      $('runCalibrationQueue')
    ) {
      $('runCalibrationQueue').disabled =
        false;
    }

    renderDatasetManager?.();
    renderCalibrationLab?.();
  }
}


/* ============================================================
   PAUSE / RESUME
============================================================ */

function toggleCalibrationPause() {
  if (
    !workerRunning
  ) {
    setWorkerUI(
      'Idle',
      'Calibration worker is not currently running.'
    );

    return;
  }

  workerPaused =
    !workerPaused;

  if (
    workerPaused
  ) {
    setWorkerUI(
      'Paused',
      'Worker will pause before the next sample.'
    );

  } else {
    setWorkerUI(
      'Running',
      'Calibration worker resumed.'
    );
  }

  if (
    $('pauseCalibrationQueue')
  ) {
    $('pauseCalibrationQueue')
      .textContent =
      workerPaused
        ? 'Resume'
        : 'Pause';
  }
}
/* ============================================================
   MAIN SCANNER
============================================================ */

async function runSmartScan() {
  const value =
    textEl?.value
      ?.trim() ||
    '';

  const words =
    wordCount(
      value
    );

  if (
    words < 80
  ) {
    alert(
      'Paste at least 80 words. 150+ words is recommended.'
    );

    return;
  }

  if (
    $('scan')
  ) {
    $('scan').disabled =
      true;
  }

  try {
    setState(
      'Analyzing…'
    );

    setProgress(
      3,
      'Profiling document…'
    );

    const analysis =
      await analyzeSample({
        value,
        forBatch:
          false
      });

    setProgress(
      95,
      'Building evidence report…'
    );

    renderScan(
      analysis
    );

    saveHistory(
      analysis
    );

    setProgress(
      100,
      'Trace complete'
    );

    if (
      isMobileDevice()
    ) {
      setState(
        'V6.4 Mobile Safe ✓'
      );

    } else if (
      analysis.consensus
        .thirdUsed
    ) {
      setState(
        'V6.4 Adaptive 3-model engine ✓'
      );

    } else {
      setState(
        'V6.4 Adaptive engine ✓'
      );
    }

    setTimeout(
      () => {
        try {
          benchmarkPrompt(
            analysis
          );

        } catch (error) {
          console.warn(
            'Benchmark prompt failed:',
            error
          );
        }
      },
      700
    );

  } catch (error) {
    console.error(
      'Smart scan failed:',
      error
    );

    setState(
      'Scan error'
    );

    alert(
      `AI Trace could not complete the scan.\n\n${error?.message || 'Unknown error'}`
    );

  } finally {
    if (
      $('scan')
    ) {
      $('scan').disabled =
        false;
    }

    hideProgress();
  }
}


/* ============================================================
   REPORT RENDER
============================================================ */

function renderScan(
  scan
) {
  const {
    consensus,
    scores,
    human,
    language,
    domain,
    domainConfidence,
    segmentScores,
    profile
  } = scan;

  $('report')
    ?.classList
    .remove(
      'hidden'
    );

  const resolved =
    consensus.verdict !==
    'INCONCLUSIVE';

  if (
    $('score')
  ) {
    $('score').textContent =
      resolved
        ? `${consensus.calibrated}%`
        : '—';
  }

  if (
    $('scaleFill')
  ) {
    $('scaleFill').style.width =
      resolved
        ? `${consensus.calibrated}%`
        : '0%';
  }

  if (
    $('verdict')
  ) {
    $('verdict').textContent =
      consensus.verdict;
  }

  const confidenceLabel =
    consensus.confidence >= 75
      ? 'High'
      : consensus.confidence >= 50
        ? 'Medium'
        : 'Low';

  if (
    $('confidence')
  ) {
    $('confidence').textContent =
      `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;
  }

  if (
    $('explain')
  ) {
    let explanation =
      `Diagnostic AI signal: ${consensus.calibrated}%. ` +
      `Evidence sufficiency: ${consensus.sufficiency.score}% (${consensus.sufficiency.level}). ` +
      `Model agreement: ${consensus.modelAgreement.agreement}%.`;

    if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {
      explanation =
        `AI Trace abstained because the available evidence was not strong enough for a reliable AI/Human attribution. ${explanation}`;
    }

    if (
      consensus.outlier.detected
    ) {
      explanation +=
        ` ${String(
          consensus.outlier.detector
        ).toUpperCase()} was down-weighted as a possible detector outlier.`;
    }

    $('explain').textContent =
      explanation;
  }

  const humanDisplay =
    clamp(
      Math.round(
        human.score * 0.72 +
        (
          100 -
          consensus.calibrated
        ) * 0.28
      )
    );

  if (
    $('humanVal')
  ) {
    $('humanVal').textContent =
      `${humanDisplay}%`;
  }

  if (
    $('aiVal')
  ) {
    $('aiVal').textContent =
      resolved
        ? `${consensus.calibrated}%`
        : 'N/A';
  }

  if (
    $('uncertainVal')
  ) {
    $('uncertainVal').textContent =
      `${consensus.uncertainty}%`;
  }

  if (
    $('humanBar')
  ) {
    $('humanBar').style.width =
      `${humanDisplay}%`;
  }

  if (
    $('aiBar')
  ) {
    $('aiBar').style.width =
      resolved
        ? `${consensus.calibrated}%`
        : '0%';
  }

  if (
    $('uncertainBar')
  ) {
    $('uncertainBar').style.width =
      `${consensus.uncertainty}%`;
  }

  if (
    $('engineBadge')
  ) {
    if (
      consensus.outlier.detected
    ) {
      $('engineBadge').textContent =
        'V6.4 • OUTLIER DEFENSE';

    } else if (
      consensus.thirdUsed
    ) {
      $('engineBadge').textContent =
        'V6.4 • ADAPTIVE 3-MODEL';

    } else {
      $('engineBadge').textContent =
        'V6.4 • ADAPTIVE CONSENSUS';
    }
  }

  const modelWeightLine =
    detector => {
      const item =
        consensus.modelWeights?.[
          detector
        ];

      if (!item) {
        return 'Inactive';
      }

      return `${item.score}% signal · weight ${item.weight}`;
    };

  const reliabilityLine =
    detector => {
      const item =
        consensus.reliability?.[
          detector
        ];

      if (!item) {
        return 'No reliability data';
      }

      return (
        `Base ${Number(
          item.base
        ).toFixed(2)} · ` +
        `AI ${Number(
          item.ai?.weight ?? 1
        ).toFixed(2)} · ` +
        `Human ${Number(
          item.human?.weight ?? 1
        ).toFixed(2)}`
      );
    };

  const humanReasons =
    human.reasons.length
      ? human.reasons
          .slice(
            0,
            5
          )
          .join(
            ' • '
          )
      : 'No strong human-style counter-signals';

  const evidence = [
    [
      'Final decision',
      consensus.verdict,
      'Outcome'
    ],

    [
      'Diagnostic AI signal',
      `${consensus.calibrated}%`,
      'Not proof'
    ],

    [
      'Adaptive raw signal',
      `${consensus.raw}%`,
      'Weighted ensemble'
    ],

    [
      'Detector median',
      `${consensus.rawMedian}%`,
      'Robust diagnostic'
    ],

    [
      'Evidence sufficiency',
      `${consensus.sufficiency.score}% — ${consensus.sufficiency.level}`,
      consensus.sufficiency.level
    ],

    [
      'Model agreement',
      `${consensus.modelAgreement.agreement}%`,
      `Spread ${consensus.modelSpread} pts`
    ],

    [
      'TMR',
      Number.isFinite(
        scores.tmr
      )
        ? `${scores.tmr}% AI signal`
        : 'Unavailable',
      modelWeightLine(
        'tmr'
      )
    ],

    [
      'TMR reliability',
      reliabilityLine(
        'tmr'
      ),
      'Adaptive'
    ],

    [
      'E5-small',
      Number.isFinite(
        scores.e5
      )
        ? `${scores.e5}% AI signal`
        : 'Unavailable',
      modelWeightLine(
        'e5'
      )
    ],

    [
      'E5 reliability',
      reliabilityLine(
        'e5'
      ),
      'Adaptive'
    ],

    [
      'ModernBERT',
      Number.isFinite(
        scores.modern
      )
        ? `${scores.modern}% AI signal`
        : 'Not used / unavailable',
      modelWeightLine(
        'modern'
      )
    ],

    [
      'Modern reliability',
      reliabilityLine(
        'modern'
      ),
      'Adaptive'
    ],

    [
      'Human counter-evidence',
      `${human.score}% — ${humanReasons}`,
      human.score >= 55
        ? 'Strong'
        : human.score >= 35
          ? 'Moderate'
          : 'Low'
    ],

    [
      'Outlier analysis',
      consensus.outlier.detected
        ? `${String(
            consensus.outlier.detector
          ).toUpperCase()} · distance ${consensus.outlier.distance} pts`
        : 'No clear detector outlier',
      consensus.outlier.detected
        ? 'Down-weighted'
        : 'Clear'
    ],

    [
      'Segment stability',
      `${consensus.segmentAnalysis.stability}%`,
      `Range ${consensus.segmentRange} pts`
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      'Context'
    ],

    [
      'Language fit',
      language === 'English'
        ? 'English — strongest supported path'
        : 'Non-English — reduced reliability',
      'Context'
    ]
  ];

  if (
    $('evidence')
  ) {
    $('evidence').innerHTML =
      evidence
        .map(
          item => `
<div class="ev">
  <div class="evTop">
    <span>${escapeHTML(item[0])}</span>
    <span>${escapeHTML(item[2])}</span>
  </div>
  <small>${escapeHTML(item[1])}</small>
</div>
`
        )
        .join('');
  }

  const metrics = {
    Words:
      profile.words,

    Sentences:
      profile.sentences,

    Domain:
      domain,

    'Domain confidence':
      domainConfidence,

    Language:
      language,

    'Models active':
      `${consensus.activeModels}/3`,

    'Sentence burstiness':
      profile
        .sentenceBurstiness
        .toFixed(2),

    'Lexical diversity':
      `${Math.round(
        profile.lexicalDiversity *
        100
      )}%`,

    'Human evidence':
      `${human.score}%`,

    'Adaptive raw signal':
      `${consensus.raw}%`,

    'Calibrated signal':
      `${consensus.calibrated}%`,

    'Evidence sufficiency':
      `${consensus.sufficiency.score}%`,

    'Evidence level':
      consensus.sufficiency.level,

    'Model agreement':
      `${consensus.modelAgreement.agreement}%`,

    'Model spread':
      `${consensus.modelSpread} pts`,

    'Model deviation':
      consensus.modelSD,

    'Segment mean':
      `${consensus.segmentAnalysis.mean}%`,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Segment stability':
      `${consensus.segmentAnalysis.stability}%`,

    Outlier:
      consensus.outlier.detected
        ? consensus.outlier.detector
        : 'None',

    Decision:
      consensus.verdict
  };

  if (
    $('metrics')
  ) {
    $('metrics').innerHTML =
      Object.entries(
        metrics
      )
        .map(
          (
            [
              key,
              value
            ]
          ) => `
<div class="metric">
  <span>${escapeHTML(key)}</span>
  <b>${escapeHTML(String(value))}</b>
</div>
`
        )
        .join('');
  }

  const chunks =
    chunkText(
      textEl?.value
        ?.trim() ||
      ''
    );

  if (
    $('segments')
  ) {
    $('segments').innerHTML =
      chunks
        .map(
          (
            chunk,
            index
          ) => {
            const score =
              segmentScores[
                index
              ] ??
              50;

            const label =
              score >= 70
                ? 'AI-supporting'
                : score <= 30
                  ? 'Human-supporting'
                  : 'Uncertain';

            return `
<div class="segment">
  <div class="segmentHead">
    <b>Segment ${index + 1}</b>
    <span>${score}% TMR · ${label}</span>
  </div>

  <div class="segmentMeter">
    <i style="width:${clamp(score)}%"></i>
  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        320
      )
    )}${chunk.length > 320 ? '…' : ''}
  </p>
</div>
`;
          }
        )
        .join('');
  }

  renderDatasetManager();

  renderCalibrationLab();

  $('report')
    ?.scrollIntoView({
      behavior:
        'smooth',

      block:
        'start'
    });
}


/* ============================================================
   BENCHMARK PROMPT
============================================================ */

function benchmarkPrompt(
  scan
) {
  const answer =
    prompt(
`AI TRACE V6.4 BENCHMARK

Only label samples whose TRUE origin you know.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known mixture
UNKNOWN = unknown origin

Cancel / leave empty to skip.`
    );

  if (!answer) {
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
      'MIXED',
      'UNKNOWN'
    ].includes(
      truth
    )
  ) {
    alert(
      'Use AI, HUMAN, MIXED or UNKNOWN.'
    );

    return;
  }

  const source =
    prompt(
      'Source / note:',
      ''
    ) || '';

  const records =
    loadBench();

  const id =
    nextBenchmarkId(
      truth,
      records
    );

  records.push({
    id,

    truth,

    source,

    savedAt:
      nowISO(),

    predictionFrozen:
      true,

    ...scan
  });

  saveBench(
    records
  );

  renderDatasetManager();

  renderCalibrationLab();

  alert(
    `Benchmark saved: ${id}`
  );
}


/* ============================================================
   DATASET SUMMARY
============================================================ */

function datasetSummary(
  records = loadBench()
) {
  const binary =
    binaryRecords(
      records
    );

  return {
    total:
      records.length,

    binary:
      binary.length,

    ai:
      records.filter(
        record =>
          record.truth === 'AI'
      ).length,

    human:
      records.filter(
        record =>
          record.truth === 'HUMAN'
      ).length,

    mixed:
      records.filter(
        record =>
          record.truth === 'MIXED'
      ).length,

    unknown:
      records.filter(
        record =>
          record.truth === 'UNKNOWN'
      ).length,

    domains:
      new Set(
        records.map(
          record =>
            record.domain ||
            'general'
        )
      ).size
  };
}


/* ============================================================
   DATASET MANAGER
============================================================ */

function renderDatasetManager() {
  const records =
    loadBench();

  const summary =
    datasetSummary(
      records
    );

  const readiness =
    benchmarkReadiness(
      records
    );

  const mapping = {
    datasetTotal:
      summary.total,

    datasetBinary:
      summary.binary,

    datasetAI:
      summary.ai,

    datasetHuman:
      summary.human,

    datasetMixed:
      summary.mixed,

    datasetUnknown:
      summary.unknown,

    datasetDomains:
      summary.domains
  };

  for (
    const [
      id,
      value
    ]
    of Object.entries(
      mapping
    )
  ) {
    if (
      $(id)
    ) {
      $(id).textContent =
        value;
    }
  }

  if (
    $('datasetStatusBadge')
  ) {
    $('datasetStatusBadge').textContent =
      readiness.level;
  }

  if (
    $('benchmarkReadinessTop')
  ) {
    $('benchmarkReadinessTop').textContent =
      `Dataset: ${readiness.level}`;
  }

  if (
    $('datasetRecords')
  ) {
    const recent =
      records
        .slice()
        .reverse()
        .slice(
          0,
          12
        );

    $('datasetRecords').innerHTML =
      recent.length
        ? recent
            .map(
              record => `
<div class="ev">
  <div class="evTop">
    <span>${escapeHTML(record.id || 'Record')}</span>
    <span>${escapeHTML(record.truth || '?')}</span>
  </div>

  <small>
    ${escapeHTML(record.domain || 'general')}
    · ${escapeHTML(record.source || 'No source')}
    · ${record.words || wordCount(record.text || '') || 0} words
  </small>
</div>
`
            )
            .join('')
        : `
<div class="ev">
  <small>No benchmark records yet.</small>
</div>
`;
  }

  const queue =
    queueSummary();

  const processed =
    queue.complete +
    queue.failed;

  if (
    !workerRunning
  ) {
    setWorkerUI(
      queue.pending
        ? 'Ready'
        : 'Idle',

      queue.total
        ? `${queue.pending} pending · ${queue.complete} complete · ${queue.failed} failed`
        : 'Import benchmark samples, then run the calibration queue.',

      queue.total
        ? processed /
          queue.total *
          100
        : 0
    );
  }
}


/* ============================================================
   DOMAIN PERFORMANCE
============================================================ */

function domainPerformance(
  records = loadBench()
) {
  const rows =
    binaryRecords(
      records
    );

  const groups =
    new Map();

  for (
    const row
    of rows
  ) {
    const domain =
      row.domain ||
      'general';

    if (
      !groups.has(
        domain
      )
    ) {
      groups.set(
        domain,
        []
      );
    }

    groups
      .get(
        domain
      )
      .push(
        row
      );
  }

  return [
    ...groups.entries()
  ]
    .map(
      (
        [
          domain,
          domainRows
        ]
      ) => ({
        domain,

        ...evaluatePredictions(
          domainRows,
          benchmarkPrediction
        )
      })
    )
    .sort(
      (
        a,
        b
      ) =>
        b.total -
        a.total
    );
}


/* ============================================================
   INSPECTORS
============================================================ */

function falsePositiveRecords(
  records = loadBench()
) {
  return binaryRecords(
    records
  )
    .filter(
      record =>
        record.truth === 'HUMAN' &&
        benchmarkPrediction(
          record
        ) === 'AI'
    );
}


function falseNegativeRecords(
  records = loadBench()
) {
  return binaryRecords(
    records
  )
    .filter(
      record =>
        record.truth === 'AI' &&
        benchmarkPrediction(
          record
        ) === 'HUMAN'
    );
}


function abstentionRecords(
  records = loadBench()
) {
  return binaryRecords(
    records
  )
    .filter(
      record =>
        benchmarkPrediction(
          record
        ) === 'ABSTAIN'
    );
}


/* ============================================================
   INSPECTOR RENDER
============================================================ */

function renderInspector(
  elementId,
  records,
  emptyText
) {
  const element =
    $(
      elementId
    );

  if (!element) {
    return;
  }

  if (
    !records.length
  ) {
    element.innerHTML = `
<div class="ev">
  <small>${escapeHTML(emptyText)}</small>
</div>
`;

    return;
  }

  element.innerHTML =
    records
      .slice()
      .reverse()
      .slice(
        0,
        10
      )
      .map(
        record => `
<div class="ev">
  <div class="evTop">
    <span>${escapeHTML(record.id || 'Record')}</span>
    <span>${escapeHTML(record.truth || '?')}</span>
  </div>

  <small>
    ${escapeHTML(record.domain || 'general')}
    · Prediction ${escapeHTML(benchmarkPrediction(record))}
    · Signal ${record.consensus?.calibrated ?? '?'}%
    · Sufficiency ${record.consensus?.sufficiency?.score ?? '?'}%
  </small>
</div>
`
      )
      .join('');
}


/* ============================================================
   CALIBRATION LAB
============================================================ */

function renderCalibrationLab() {
  const records =
    loadBench();

  const binary =
    binaryRecords(
      records
    );

  const readiness =
    benchmarkReadiness(
      records
    );

  const ensemble =
    evaluatePredictions(
      binary,
      benchmarkPrediction
    );

  if (
    $('calibrationStatusBadge')
  ) {
    $('calibrationStatusBadge').textContent =
      readiness.level;
  }

  if (
    $('calibrationReadiness')
  ) {
    $('calibrationReadiness').innerHTML = `
<div class="ev">
  <div class="evTop">
    <span>${readiness.level}</span>
    <span>
      ${
        readiness.level === 'COLLECTING'
          ? 'LEARNING LIMITED'
          : 'ADAPTIVE LEARNING ACTIVE'
      }
    </span>
  </div>

  <small>
    ${readiness.total} binary samples
    · ${readiness.ai} AI
    · ${readiness.human} HUMAN
    · ${readiness.domains} domains
  </small>
</div>
`;
  }

  if (
    $('ensembleMetrics')
  ) {
    const metrics = {
      'Binary samples':
        ensemble.total,

      Coverage:
        `${ensemble.coverage}%`,

      'Selective accuracy':
        `${ensemble.selectiveAccuracy}%`,

      Precision:
        `${ensemble.precision}%`,

      'AI recall':
        `${ensemble.recall}%`,

      'Human specificity':
        `${ensemble.specificity}%`,

      'False-positive rate':
        `${ensemble.fpr}%`,

      'False-negative rate':
        `${ensemble.fnr}%`,

      'AI abstention rate':
        `${ensemble.aiAbstainRate}%`,

      'Human abstention rate':
        `${ensemble.humanAbstainRate}%`
    };

    $('ensembleMetrics').innerHTML =
      Object.entries(
        metrics
      )
        .map(
          (
            [
              key,
              value
            ]
          ) => `
<div class="metric">
  <span>${escapeHTML(key)}</span>
  <b>${escapeHTML(String(value))}</b>
</div>
`
        )
        .join('');
  }

  if (
    $('detectorReliability')
  ) {
    const reliability =
      buildModelReliability(
        'general',
        records
      );

    $('detectorReliability').innerHTML =
      [
        'tmr',
        'e5',
        'modern'
      ]
        .map(
          detector => {
            const item =
              reliability[
                detector
              ];

            const metrics =
              item.global.metrics;

            return `
<div class="ev">
  <div class="evTop">
    <span>${escapeHTML(detector.toUpperCase())}</span>
    <span>${item.global.samples} samples</span>
  </div>

  <small>
    Base reliability ${Number(item.base).toFixed(2)}
    · AI weight ${Number(item.ai.weight).toFixed(2)}
    · Human weight ${Number(item.human.weight).toFixed(2)}
    · Accuracy ${metrics.selectiveAccuracy}%
    · Coverage ${metrics.coverage}%
    · FPR ${metrics.fpr}%
    · FNR ${metrics.fnr}%
  </small>
</div>
`;
          }
        )
        .join('');
  }

  if (
    $('domainPerformance')
  ) {
    const domains =
      domainPerformance(
        records
      );

    $('domainPerformance').innerHTML =
      domains.length
        ? domains
            .map(
              item => `
<div class="metric">
  <span>${escapeHTML(item.domain)}</span>
  <b>
    ${item.total} samples
    · Acc ${item.selectiveAccuracy}%
    · Coverage ${item.coverage}%
    · FPR ${item.fpr}%
    · FNR ${item.fnr}%
  </b>
</div>
`
            )
            .join('')
        : `
<div class="metric">
  <span>No domain data</span>
  <b>—</b>
</div>
`;
  }

  renderInspector(
    'falsePositiveInspector',
    falsePositiveRecords(
      records
    ),
    'No known HUMAN sample has been classified as AI.'
  );

  renderInspector(
    'falseNegativeInspector',
    falseNegativeRecords(
      records
    ),
    'No known AI sample has been classified as HUMAN.'
  );

  renderInspector(
    'abstentionInspector',
    abstentionRecords(
      records
    ),
    'No binary abstentions recorded.'
  );
}


/* ============================================================
   BULK IMPORT PARSER
============================================================ */

function parseBulkImport(
  raw
) {
  const lines =
    String(
      raw || ''
    )
      .split(
        /\n/
      )
      .map(
        line =>
          line.trim()
      )
      .filter(Boolean);

  const samples = [];
  const errors = [];

  lines.forEach(
    (
      line,
      index
    ) => {
      try {
        const item =
          JSON.parse(
            line
          );

        const truth =
          String(
            item.truth ||
            ''
          )
            .trim()
            .toUpperCase();

        if (
          ![
            'AI',
            'HUMAN',
            'MIXED',
            'UNKNOWN'
          ].includes(
            truth
          )
        ) {
          throw new Error(
            'Invalid truth label'
          );
        }

        const sampleText =
          String(
            item.text ||
            ''
          )
            .trim();

        if (
          wordCount(
            sampleText
          ) < 30
        ) {
          throw new Error(
            'Sample text must contain at least 30 words'
          );
        }

        samples.push({
          truth,

          source:
            String(
              item.source ||
              ''
            ),

          domain:
            String(
              item.domain ||
              'auto'
            ),

          text:
            sampleText
        });

      } catch (error) {
        errors.push(
          `Line ${index + 1}: ${error.message}`
        );
      }
    }
  );

  return {
    samples,
    errors
  };
}


/* ============================================================
   BULK IMPORT VALIDATION
============================================================ */

function validateBulkImport() {
  const raw =
    $('bulkImportText')
      ?.value ||
    '';

  const parsed =
    parseBulkImport(
      raw
    );

  if (
    $('bulkImportCount')
  ) {
    $('bulkImportCount').textContent =
      `${parsed.samples.length} samples detected`;
  }

  if (
    $('bulkImportResult')
  ) {
    $('bulkImportResult')
      .classList
      .remove(
        'hidden'
      );

    $('bulkImportResult').innerHTML =
      parsed.errors.length
        ? `
<b>${parsed.samples.length} valid samples.</b><br><br>
${parsed.errors
  .map(
    error =>
      escapeHTML(
        error
      )
  )
  .join(
    '<br>'
  )}
`
        : `
<b>${parsed.samples.length} valid samples.</b><br>
No formatting errors detected.
`;
  }

  return parsed;
}


/* ============================================================
   BULK IMPORT
============================================================ */

function importBulkSamples() {
  const parsed =
    validateBulkImport();

  if (
    !parsed.samples.length
  ) {
    alert(
      'No valid samples to import.'
    );

    return;
  }

  const queue =
    loadCalibrationQueue();

  const imported =
    parsed.samples.map(
      sample => ({
        importId:
          `IMP-${Date.now()}-${Math.random()
            .toString(36)
            .slice(2, 9)}`,

        status:
          'PENDING',

        createdAt:
          nowISO(),

        attempts:
          0,

        ...sample
      })
    );

  saveCalibrationQueue([
    ...queue,
    ...imported
  ]);

  if (
    $('bulkImportResult')
  ) {
    $('bulkImportResult').innerHTML =
      `<b>${imported.length} samples added to the calibration queue.</b><br>
       Press Run Calibration Queue to analyze them automatically.`;
  }

  if (
    $('bulkImportText')
  ) {
    $('bulkImportText').value =
      '';
  }

  if (
    $('bulkImportCount')
  ) {
    $('bulkImportCount').textContent =
      '0 samples detected';
  }

  renderDatasetManager();

  alert(
    `${imported.length} samples imported successfully.`
  );
}


/* ============================================================
   JSON EXPORT
============================================================ */

function exportBenchmarkJSON() {
  const records =
    loadBench();

  const payload = {
    version:
      VERSION,

    exportedAt:
      nowISO(),

    readiness:
      benchmarkReadiness(
        records
      ),

    summary:
      datasetSummary(
        records
      ),

    ensemble:
      evaluatePredictions(
        binaryRecords(
          records
        ),
        benchmarkPrediction
      ),

    domains:
      domainPerformance(
        records
      ),

    queue:
      queueSummary(),

    records
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

  const anchor =
    document.createElement(
      'a'
    );

  anchor.href =
    url;

  anchor.download =
    `AI-Trace-V64-Benchmark-${Date.now()}.json`;

  document.body
    .appendChild(
      anchor
    );

  anchor.click();

  anchor.remove();

  URL.revokeObjectURL(
    url
  );
}


/* ============================================================
   CSV EXPORT
============================================================ */

function exportBenchmarkCSV() {
  const records =
    loadBench();

  const headers = [
    'id',
    'truth',
    'source',
    'version',
    'timestamp',
    'domain',
    'language',
    'words',
    'tmr',
    'e5',
    'modern',
    'humanEvidence',
    'rawSignal',
    'calibratedSignal',
    'sufficiency',
    'confidence',
    'uncertainty',
    'verdict',
    'modelAgreement',
    'modelSpread',
    'segmentRange'
  ];

  const csvEscape =
    value => {
      const text =
        String(
          value ??
          ''
        );

      if (
        /[",\n]/.test(
          text
        )
      ) {
        return `"${text.replace(
          /"/g,
          '""'
        )}"`;
      }

      return text;
    };

  const rows =
    records.map(
      record => [
        record.id,
        record.truth,
        record.source,
        record.version,
        record.timestamp ||
          record.savedAt,
        record.domain,
        record.language,
        record.words,
        record.scores?.tmr,
        record.scores?.e5,
        record.scores?.modern,
        record.human?.score,
        record.consensus?.raw,
        record.consensus?.calibrated,
        record.consensus?.sufficiency?.score,
        record.consensus?.confidence,
        record.consensus?.uncertainty,
        record.consensus?.verdict,
        record.consensus?.modelAgreement?.agreement,
        record.consensus?.modelSpread,
        record.consensus?.segmentRange
      ]
        .map(
          csvEscape
        )
        .join(',')
    );

  const csv =
    [
      headers.join(','),
      ...rows
    ].join(
      '\n'
    );

  const blob =
    new Blob(
      [csv],
      {
        type:
          'text/csv;charset=utf-8'
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
    `AI-Trace-V64-Benchmark-${Date.now()}.csv`;

  document.body
    .appendChild(
      anchor
    );

  anchor.click();

  anchor.remove();

  URL.revokeObjectURL(
    url
  );
}


/* ============================================================
   HISTORY
============================================================ */

function saveHistory(
  scan
) {
  const history =
    loadJSON(
      HISTORY_KEY,
      []
    );

  history.push({
    timestamp:
      scan.timestamp,

    version:
      VERSION,

    words:
      scan.words,

    language:
      scan.language,

    domain:
      scan.domain,

    scores:
      scan.scores,

    humanScore:
      scan.human.score,

    consensus:
      scan.consensus
  });

  saveJSON(
    HISTORY_KEY,
    history.slice(
      -100
    )
  );
}


/* ============================================================
   EVENT LISTENERS
============================================================ */

textEl
  ?.addEventListener(
    'input',
    updateCount
  );


$('demo')
  ?.addEventListener(
    'click',
    loadDemo
  );


$('clear')
  ?.addEventListener(
    'click',
    () => {
      if (
        textEl
      ) {
        textEl.value =
          '';
      }

      updateCount();

      $('report')
        ?.classList
        .add(
          'hidden'
        );
    }
  );


$('scan')
  ?.addEventListener(
    'click',
    runSmartScan
  );


$('openBulkImport')
  ?.addEventListener(
    'click',
    () => {
      $('bulkImportPanel')
        ?.classList
        .remove(
          'hidden'
        );

      $('bulkImportPanel')
        ?.scrollIntoView({
          behavior:
            'smooth',

          block:
            'start'
        });
    }
  );


$('closeBulkImport')
  ?.addEventListener(
    'click',
    () => {
      $('bulkImportPanel')
        ?.classList
        .add(
          'hidden'
        );
    }
  );


$('bulkImportText')
  ?.addEventListener(
    'input',
    () => {
      const parsed =
        parseBulkImport(
          $('bulkImportText')
            ?.value ||
          ''
        );

      if (
        $('bulkImportCount')
      ) {
        $('bulkImportCount').textContent =
          `${parsed.samples.length} samples detected`;
      }
    }
  );


$('validateBulkImport')
  ?.addEventListener(
    'click',
    validateBulkImport
  );


$('importBulkSamples')
  ?.addEventListener(
    'click',
    importBulkSamples
  );


$('runCalibrationQueue')
  ?.addEventListener(
    'click',
    () => {
      if (
        workerRunning &&
        workerPaused
      ) {
        workerPaused =
          false;

        if (
          $('pauseCalibrationQueue')
        ) {
          $('pauseCalibrationQueue').textContent =
            'Pause';
        }

        setWorkerUI(
          'Running',
          'Calibration worker resumed.'
        );

        return;
      }

      runCalibrationWorker();
    }
  );


$('pauseCalibrationQueue')
  ?.addEventListener(
    'click',
    toggleCalibrationPause
  );


$('exportBenchmarkJSON')
  ?.addEventListener(
    'click',
    exportBenchmarkJSON
  );


$('exportBenchmarkCSV')
  ?.addEventListener(
    'click',
    exportBenchmarkCSV
  );


$('clearBenchmark')
  ?.addEventListener(
    'click',
    () => {
      const confirmation =
        confirm(
          'Delete all V6.4 benchmark records from this browser?'
        );

      if (
        !confirmation
      ) {
        return;
      }

      localStorage.removeItem(
        BENCH_KEY
      );

      renderDatasetManager();

      renderCalibrationLab();

      alert(
        'V6.4 benchmark dataset deleted.'
      );
    }
  );


/* ============================================================
   DEVELOPER API
============================================================ */

window.AITraceV64 = {
  report() {
    const records =
      loadBench();

    return {
      version:
        VERSION,

      summary:
        datasetSummary(
          records
        ),

      readiness:
        benchmarkReadiness(
          records
        ),

      metrics:
        evaluatePredictions(
          binaryRecords(
            records
          ),
          benchmarkPrediction
        ),

      domains:
        domainPerformance(
          records
        ),

      queue:
        queueSummary(),

      records
    };
  },

  reliability(
    domain = 'general'
  ) {
    return buildModelReliability(
      domain,
      loadBench()
    );
  },

  queue() {
    return loadCalibrationQueue();
  },

  history() {
    return loadJSON(
      HISTORY_KEY,
      []
    );
  },

  runQueue() {
    return runCalibrationWorker();
  },

  pauseQueue() {
    workerPaused =
      true;

    setWorkerUI(
      'Paused',
      'Worker will pause before the next sample.'
    );
  },

  resumeQueue() {
    workerPaused =
      false;

    setWorkerUI(
      'Running',
      'Worker resumed.'
    );
  },

  stopQueue() {
    workerAbortRequested =
      true;

    workerPaused =
      false;

    setWorkerUI(
      'Stopping',
      'Worker will stop after the current sample.'
    );
  },

  retryFailed() {
    const queue =
      loadCalibrationQueue();

    for (
      const item
      of queue
    ) {
      if (
        item.status === 'FAILED'
      ) {
        item.status =
          'PENDING';

        item.error =
          null;
      }
    }

    saveCalibrationQueue(
      queue
    );

    renderDatasetManager();
  },

  exportJSON() {
    exportBenchmarkJSON();
  },

  exportCSV() {
    exportBenchmarkCSV();
  },

  clearQueue() {
    const confirmation =
      confirm(
        'Delete the entire calibration queue?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      IMPORT_KEY
    );

    renderDatasetManager();
  },

  clearBenchmark() {
    const confirmation =
      confirm(
        'Delete the full benchmark dataset?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      BENCH_KEY
    );

    renderDatasetManager();

    renderCalibrationLab();
  },

  clearHistory() {
    localStorage.removeItem(
      HISTORY_KEY
    );
  }
};


/* ============================================================
   INITIALIZATION
============================================================ */

recoverInterruptedQueue();

updateCount();

renderDatasetManager();

renderCalibrationLab();


const initialQueue =
  queueSummary();


if (
  initialQueue.total
) {
  const processed =
    initialQueue.complete +
    initialQueue.failed;

  setWorkerUI(
    initialQueue.pending
      ? 'Ready'
      : 'Complete',

    `${initialQueue.pending} pending · ${initialQueue.complete} complete · ${initialQueue.failed} failed`,

    initialQueue.total
      ? processed /
        initialQueue.total *
        100
      : 0
  );
}


console.info(
  `AI TRACE V${VERSION} loaded successfully`
);
