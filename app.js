/*
  ============================================================
  AI TRACE V6.3 — CALIBRATION DATASET ENGINE
  COMPLETE APP.JS
  ============================================================

  Features
  ------------------------------------------------------------
  - TMR detector
  - E5-small detector
  - Conditional ModernBERT judge
  - Segment analysis
  - Human counter-evidence
  - Domain estimation
  - Conservative abstention
  - Adaptive detector reliability
  - Leave-one-out benchmark reliability
  - Bulk benchmark import
  - Dataset manager
  - Calibration dashboard
  - False-positive inspector
  - False-negative inspector
  - Abstention inspector
  - JSON / CSV export
  - Zero paid API
*/


import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';


/* ============================================================
   TRANSFORMERS CONFIG
============================================================ */

env.allowLocalModels = false;
env.useBrowserCache = true;


/* ============================================================
   VERSION / MODELS
============================================================ */

const VERSION = '6.3';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';


/* ============================================================
   STORAGE
============================================================ */

const BENCH_KEY =
  'aiTraceBenchmarkV63';

const HISTORY_KEY =
  'aiTraceHistoryV63';

const IMPORT_KEY =
  'aiTraceImportedDatasetV63';


/* ============================================================
   MODEL INSTANCES
============================================================ */

let tmrModel = null;
let e5Model = null;
let modernModel = null;


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


function wordCount(
  value
) {

  if (
    !String(value || '')
      .trim()
  ) {

    return 0;
  }

  return String(value)
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .length;
}


function average(
  values
) {

  const usable =
    values.filter(
      Number.isFinite
    );

  if (
    !usable.length
  ) {

    return 0;
  }

  return usable.reduce(
    (
      total,
      value
    ) =>
      total + value,
    0
  ) / usable.length;
}


function median(
  values
) {

  const usable =
    values
      .filter(
        Number.isFinite
      )
      .sort(
        (
          a,
          b
        ) =>
          a - b
      );

  if (
    !usable.length
  ) {

    return 50;
  }

  const middle =
    Math.floor(
      usable.length /
      2
    );

  if (
    usable.length %
    2
  ) {

    return usable[
      middle
    ];
  }

  return (
    usable[
      middle - 1
    ] +
    usable[
      middle
    ]
  ) / 2;
}


function standardDeviation(
  values
) {

  const usable =
    values.filter(
      Number.isFinite
    );

  if (
    !usable.length
  ) {

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


function safePercent(
  numerator,
  denominator
) {

  if (
    !denominator
  ) {

    return 0;
  }

  return Math.round(
    numerator /
    denominator *
    100
  );
}


function escapeHTML(
  value
) {

  return String(
    value ?? ''
  ).replace(
    /[&<>"']/g,
    character =>
      ({
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
      })[
        character
      ]
  );
}


function loadJSON(
  key,
  fallback = []
) {

  try {

    const raw =
      localStorage.getItem(
        key
      );

    if (
      !raw
    ) {

      return fallback;
    }

    return JSON.parse(
      raw
    );

  } catch (
    error
  ) {

    console.warn(
      'Storage read failed:',
      key,
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

  } catch (
    error
  ) {

    console.warn(
      'Storage write failed:',
      key,
      error
    );

    return false;
  }
}


function detectLanguage(
  value
) {

  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) ||
      []
    ).length;

  const allLetters =
    (
      value.match(
        /\p{L}/gu
      ) ||
      []
    ).length;

  if (
    !allLetters
  ) {

    return 'Unknown';
  }

  return (
    latin /
    allLetters >=
    0.82
  )
    ? 'English'
    : 'Non-English';
}


function countMatches(
  value,
  regex
) {

  return (
    value.match(
      regex
    ) ||
    []
  ).length;
}


function isMobileDevice() {

  return (
    window.matchMedia(
      '(max-width: 768px)'
    ).matches ||

    /Android|iPhone|iPad|iPod/i
      .test(
        navigator.userAgent
      )
  );
}


/* ============================================================
   UI HELPERS
============================================================ */

function setProgress(
  percent,
  label
) {

  $('progress')
    ?.classList
    .remove(
      'hidden'
    );

  if (
    $('bar')
  ) {

    $('bar').style.width =
      `${clamp(
        percent
      )}%`;
  }

  if (
    $('progressText')
  ) {

    $('progressText')
      .textContent =
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


function setState(
  value
) {

  if (
    $('modelState')
  ) {

    $('modelState')
      .textContent =
      value;
  }
}


function updateCount() {

  if (
    !textEl ||
    !$('count')
  ) {

    return;
  }

  $('count').textContent =
    `${wordCount(
      textEl.value
    )} words`;
}


/* ============================================================
   DEMO
============================================================ */

function loadDemo() {

  if (
    !textEl
  ) {

    return;
  }

  textEl.value =
`Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

One major advantage of artificial intelligence is its ability to process information at a scale that would be difficult for humans to match. Organizations can use automated systems to identify patterns in large datasets, detect anomalies, and improve operational efficiency.

However, artificial intelligence also introduces new challenges. Machine-generated content may contain factual errors, misleading statements, or fabricated information. As generated media becomes more realistic, determining the origin of digital content becomes increasingly difficult.

Reliable AI detection will therefore depend on careful evaluation, transparent limitations, and continuous testing across many different types of content. A useful detection system should provide evidence rather than absolute claims and should acknowledge uncertainty whenever the available signals disagree.`;

  updateCount();
}


/* ============================================================
   DOCUMENT PROFILE
============================================================ */

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
      .split(
        /[.!?]+/
      )
      .map(
        sentence =>
          sentence.trim()
      )
      .filter(Boolean);

  const paragraphs =
    value
      .split(
        /\n\s*\n/
      )
      .map(
        paragraph =>
          paragraph.trim()
      )
      .filter(Boolean);

  const lines =
    value
      .split(
        /\n/
      )
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
            ) ||
            []
          ).length >
          0
      )
      .length;

  const quoteCount =
    (
      value.match(
        /["“”‘’]/g
      ) ||
      []
    ).length;

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
        /^[“"'—-]/
          .test(
            line
          ) ||
        /[”"']$/
          .test(
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
        (
          a,
          b
        ) =>
          b[1] -
          a[1]
      );

  const [
    domain,
    score
  ] =
    sorted[0] ||
    [
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
    profile.sentenceBurstiness >=
    0.70
  ) {

    score += 20;

    reasons.push(
      'High sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score += 12;

    reasons.push(
      'Moderate sentence-length variation'
    );
  }


  if (
    profile.punctuationTypes >=
    5
  ) {

    score += 12;

    reasons.push(
      'Rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >=
    3
  ) {

    score += 6;
  }


  if (
    profile.firstPerson >=
    4
  ) {

    score += 10;

    reasons.push(
      'Personal or subjective voice'
    );

  } else if (
    profile.firstPerson >
    0
  ) {

    score += 5;
  }


  if (
    profile.contractions >=
    3
  ) {

    score += 8;

    reasons.push(
      'Natural contraction usage'
    );
  }


  if (
    profile.quoteCount >=
    6 ||
    profile.dialogueLines >=
    2
  ) {

    score += 14;

    reasons.push(
      'Dialogue or quotation structure'
    );
  }


  if (
    profile.paragraphDeviation >=
    18 &&
    profile.paragraphs >=
    3
  ) {

    score += 8;

    reasons.push(
      'Irregular paragraph rhythm'
    );
  }


  if (
    profile.transitions >=
    4
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
    ) ||
    [
      value
    ];

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
   MODEL LOADING
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
        : [
            output
          ]
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
      label.includes(
        'ai'
      ) ||
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
      label.includes(
        'human'
      ) ||
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
    results.length >=
    2
  ) {

    return Number(
      results[
        1
      ]?.score ??
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
    ) *
    100
  );
}


/* ============================================================
   BENCHMARK STORAGE
============================================================ */

function loadBench() {

  return loadJSON(
    BENCH_KEY,
    []
  );
}


function saveBench(
  records
) {

  return saveJSON(
    BENCH_KEY,
    records
  );
}


function binaryRecords(
  records =
    loadBench()
) {

  return records.filter(
    record =>
      record.truth ===
        'AI' ||
      record.truth ===
        'HUMAN'
  );
}


/* ============================================================
   PREDICTIONS
============================================================ */

function benchmarkPrediction(
  record
) {

  const verdict =
    record.consensus
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
   GENERAL METRIC EVALUATOR
============================================================ */

function evaluatePredictions(
  rows,
  predictionFunction
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

    const prediction =
      predictionFunction(
        row
      );

    if (
      prediction ===
      'ABSTAIN'
    ) {

      if (
        row.truth ===
        'AI'
      ) {

        aiAbstain++;

      } else if (
        row.truth ===
        'HUMAN'
      ) {

        humanAbstain++;
      }

      continue;
    }

    if (
      row.truth ===
        'AI' &&
      prediction ===
        'AI'
    ) {

      TP++;
    }

    if (
      row.truth ===
        'HUMAN' &&
      prediction ===
        'HUMAN'
    ) {

      TN++;
    }

    if (
      row.truth ===
        'HUMAN' &&
      prediction ===
        'AI'
    ) {

      FP++;
    }

    if (
      row.truth ===
        'AI' &&
      prediction ===
        'HUMAN'
    ) {

      FN++;
    }
  }

  const totalAI =
    rows.filter(
      row =>
        row.truth ===
        'AI'
    ).length;

  const totalHuman =
    rows.filter(
      row =>
        row.truth ===
        'HUMAN'
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

    decided,

    abstentions:
      aiAbstain +
      humanAbstain,

    TP,
    TN,
    FP,
    FN,

    aiAbstain,
    humanAbstain,

    coverage:
      safePercent(
        decided,
        rows.length
      ),

    selectiveAccuracy:
      safePercent(
        TP + TN,
        decided
      ),

    precision:
      safePercent(
        TP,
        TP + FP
      ),

    recall:
      safePercent(
        TP,
        totalAI
      ),

    specificity:
      safePercent(
        TN,
        totalHuman
      ),

    fpr:
      safePercent(
        FP,
        totalHuman
      ),

    fnr:
      safePercent(
        FN,
        totalAI
      ),

    aiAbstainRate:
      safePercent(
        aiAbstain,
        totalAI
      ),

    humanAbstainRate:
      safePercent(
        humanAbstain,
        totalHuman
      )
  };
}


/* ============================================================
   DETECTOR-SPECIFIC PREDICTION
============================================================ */

function detectorPrediction(
  record,
  detector
) {

  const value =
    record.scores?.[
      detector
    ];

  if (
    !Number.isFinite(
      value
    )
  ) {

    return 'ABSTAIN';
  }

  if (
    value >=
    70
  ) {

    return 'AI';
  }

  if (
    value <=
    30
  ) {

    return 'HUMAN';
  }

  return 'ABSTAIN';
}


/* ============================================================
   LEAVE-ONE-OUT RECORD FILTER
============================================================ */

function leaveOneOutRecords(
  records,
  excludeId
) {

  if (
    !excludeId
  ) {

    return records;
  }

  return records.filter(
    record =>
      record.id !==
      excludeId
  );
}


/* ============================================================
   DETECTOR RELIABILITY
============================================================ */

function detectorReliabilityMetrics(
  detector,
  domain,
  records,
  excludeId = null
) {

  const cleanRecords =
    leaveOneOutRecords(
      binaryRecords(
        records
      ),
      excludeId
    );

  const globalRows =
    cleanRecords.filter(
      record =>
        Number.isFinite(
          record.scores?.[
            detector
          ]
        )
    );

  const domainRows =
    globalRows.filter(
      record =>
        (
          record.domain ||
          'general'
        ) ===
        domain
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

    global:
      globalMetrics,

    domain:
      domainMetrics,

    globalSamples:
      globalRows.length,

    domainSamples:
      domainRows.length
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
    Small-sample protection:
    below 20 samples reliability stays close to neutral.
  */

  if (
    sampleCount <
    20
  ) {

    return 1;
  }


  const accuracy =
    metrics.selectiveAccuracy /
    100;

  const coverage =
    metrics.coverage /
    100;

  const fpr =
    metrics.fpr /
    100;

  const fnr =
    metrics.fnr /
    100;


  let score =
    0.50 +
    accuracy * 0.45 +
    coverage * 0.15 -
    fpr * 0.40 -
    fnr * 0.25;


  /*
    Smooth confidence ramp.
  */

  const maturity =
    clamp(
      sampleCount /
      100,
      0,
      1
    );


  score =
    1 +
    (
      score -
      1
    ) *
    maturity;


  return clamp(
    score,
    0.35,
    1.35
  );
}


/* ============================================================
   DIRECTIONAL RELIABILITY
============================================================ */

function directionalWeight(
  detector,
  domain,
  direction,
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


  /*
    Prefer domain data only after enough domain samples.
  */

  const useDomain =
    stats.domainSamples >=
    20;


  const metrics =
    useDomain
      ? stats.domain
      : stats.global;


  const samples =
    useDomain
      ? stats.domainSamples
      : stats.globalSamples;


  if (
    samples <
    20
  ) {

    return 1;
  }


  let weight =
    calculateReliabilityWeight(
      metrics,
      samples
    );


  /*
    Additional asymmetric protection.

    AI direction:
    high FPR reduces AI trust strongly.

    HUMAN direction:
    high FNR reduces HUMAN trust.
  */

  if (
    direction ===
    'AI'
  ) {

    weight *=
      1 -
      (
        metrics.fpr /
        100
      ) *
      0.65;

  } else {

    weight *=
      1 -
      (
        metrics.fnr /
        100
      ) *
      0.65;
  }


  return clamp(
    weight,
    0.30,
    1.35
  );
}


/* ============================================================
   RELIABILITY OBJECT
============================================================ */

function buildModelReliability(
  domain,
  records =
    loadBench(),
  excludeId =
    null
) {

  const detectors = [
    'tmr',
    'e5',
    'modern'
  ];

  const result = {};

  for (
    const detector
    of detectors
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
        stats.global,
        stats.globalSamples
      );

    result[
      detector
    ] = {

      base,

      global:
        {
          samples:
            stats.globalSamples,

          metrics:
            stats.global
        },

      domain:
        {
          samples:
            stats.domainSamples,

          metrics:
            stats.domain
        },

      ai:
        {
          weight:
            directionalWeight(
              detector,
              domain,
              'AI',
              records,
              excludeId
            )
        },

      human:
        {
          weight:
            directionalWeight(
              detector,
              domain,
              'HUMAN',
              records,
              excludeId
            )
        }
    };
  }

  return result;
}


/* ============================================================
   THIRD MODEL ROUTING
============================================================ */

function shouldUseThirdModel({
  scores,
  human,
  segmentScores,
  domain,
  words
}) {

  if (
    !Number.isFinite(
      scores.tmr
    ) ||
    !Number.isFinite(
      scores.e5
    )
  ) {

    return false;
  }

  const gap =
    Math.abs(
      scores.tmr -
      scores.e5
    );

  const quickMedian =
    median(
      [
        scores.tmr,
        scores.e5
      ]
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

  return (
    words < 180 ||

    gap >= 18 ||

    (
      quickMedian >=
        35 &&
      quickMedian <=
        85
    ) ||

    segmentRange >=
      40 ||

    human.score >=
      35 ||

    domain ===
      'books' ||

    domain ===
      'poetry'
  );
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
    entries.length <
    3
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

  const med =
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
            med
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
    first.distance >=
      28 &&
    (
      first.distance -
      second.distance
    ) >=
      12;

  return {

    detected,

    detector:
      detected
        ? first.detector
        : null,

    distance:
      Math.round(
        first.distance
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
        score >=
        70
    ).length;

  const humanSegments =
    valid.filter(
      score =>
        score <=
        30
    ).length;

  const uncertainSegments =
    valid.length -
    aiSegments -
    humanSegments;

  const stability =
    clamp(
      Math.round(
        100 -
        deviation * 1.25 -
        range * 0.35
      )
    );

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

    stability,

    aiSegments,

    humanSegments,

    uncertainSegments,

    mixed:
      aiSegments >
        0 &&
      humanSegments >
        0
  };
}


/* ============================================================
   CALIBRATION READINESS
============================================================ */

function benchmarkReadiness(
  records =
    loadBench()
) {

  const rows =
    binaryRecords(
      records
    );

  const ai =
    rows.filter(
      record =>
        record.truth ===
        'AI'
    ).length;

  const human =
    rows.filter(
      record =>
        record.truth ===
        'HUMAN'
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
    values.length <=
    1
  ) {

    return {

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
  activeModels,
  agreement,
  segmentAnalysis,
  outlier,
  human,
  domain
}) {

  let score = 100;


  if (
    profile.words <
    100
  ) {

    score -= 30;

  } else if (
    profile.words <
    150
  ) {

    score -= 18;

  } else if (
    profile.words <
    220
  ) {

    score -= 8;
  }


  if (
    language !==
    'English'
  ) {

    score -= 35;
  }


  if (
    activeModels ===
    1
  ) {

    score -= 35;

  } else if (
    activeModels ===
    2
  ) {

    score -= 10;
  }


  score -=
    Math.round(
      (
        100 -
        agreement.agreement
      ) *
      0.35
    );


  score -=
    Math.round(
      (
        100 -
        segmentAnalysis.stability
      ) *
      0.20
    );


  if (
    outlier.detected
  ) {

    score -= 5;
  }


  if (
    domain ===
      'books' ||
    domain ===
      'poetry'
  ) {

    score -= 10;
  }


  if (
    human.score >=
    55
  ) {

    score -= 5;
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
    score >=
    75
  ) {

    level =
      'STRONG';

  } else if (
    score >=
    55
  ) {

    level =
      'MODERATE';

  } else if (
    score >=
    40
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
   ADAPTIVE CONSENSUS
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

  const activeEntries =
    Object.entries(
      scores
    )
      .filter(
        (
          [
            ,
            score
          ]
        ) =>
          Number.isFinite(
            score
          )
      );


  const activeScores =
    activeEntries.map(
      (
        [
          ,
          score
        ]
      ) =>
        score
    );


  const activeModels =
    activeScores.length;


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


  const modelWeights = {};

  let weightedTotal = 0;
  let weightTotal = 0;


  for (
    const [
      detector,
      score
    ]
    of activeEntries
  ) {

    const direction =
      score >=
        50
        ? 'ai'
        : 'human';


    let weight =
      reliability[
        detector
      ]?.[
        direction
      ]?.weight ??
      1;


    /*
      Soft outlier penalty.
    */

    if (
      outlier.detected &&
      outlier.detector ===
      detector
    ) {

      weight *= 0.45;
    }


    /*
      Do not allow a tiny benchmark to create extreme weights.
    */

    const readiness =
      benchmarkReadiness(
        benchmarkRecords
      );


    if (
      readiness.level ===
      'COLLECTING'
    ) {

      weight =
        Math.max(
          0.85,
          Math.min(
            1.10,
            weight
          )
        );
    }


    weight =
      clamp(
        weight,
        0.25,
        1.40
      );


    modelWeights[
      detector
    ] = {

      score,

      weight:
        Number(
          weight.toFixed(
            3
          )
        )
    };


    weightedTotal +=
      score *
      weight;


    weightTotal +=
      weight;
  }


  const weightedRaw =
    weightTotal
      ? weightedTotal /
        weightTotal
      : 50;


  const rawMedian =
    median(
      activeScores
    );


  /*
    Blend weighted signal with robust median.
  */

  const raw =
    clamp(
      Math.round(
        weightedRaw *
        0.62 +
        rawMedian *
        0.38
      )
    );


  const modelAgreement =
    calculateModelAgreement(
      scores
    );


  const segmentAnalysis =
    analyzeSegments(
      segmentScores
    );


  const sufficiency =
    calculateEvidenceSufficiency({

      profile,

      language,

      activeModels,

      agreement:
        modelAgreement,

      segmentAnalysis,

      outlier,

      human,

      domain
    });


  /*
    Human counter-evidence reduces confidence in AI,
    especially under model disagreement.
  */

  const disagreement =
    1 -
    modelAgreement.agreement /
    100;


  const humanPenalty =
    human.score *
    (
      0.08 +
      disagreement *
      0.35
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
    Protected literary domains.
  */

  if (
    (
      domain ===
        'books' ||
      domain ===
        'poetry'
    ) &&
    human.score >=
      40 &&
    modelAgreement.agreement <
      55
  ) {

    calibrated =
      Math.min(
        calibrated,
        64
      );
  }


  let uncertainty =
    clamp(
      Math.round(
        100 -
        sufficiency.score
      ),
      5,
      95
    );


  /*
    Conflicting high-AI + human evidence.
  */

  if (
    calibrated >=
      70 &&
    human.score >=
      50
  ) {

    uncertainty =
      clamp(
        uncertainty +
        Math.round(
          (
            human.score -
            45
          ) *
          0.30
        ),
        5,
        95
      );
  }


  const confidence =
    100 -
    uncertainty;


  const highConflict =
    modelAgreement.spread >=
      45 ||
    modelAgreement.agreement <=
      30;


  const severeConflict =
    modelAgreement.spread >=
      70;


  let verdict =
    'INCONCLUSIVE';


  /*
    Conservative AI classification.
  */

  if (
    language ===
      'English' &&

    activeModels >=
      2 &&

    calibrated >=
      86 &&

    sufficiency.score >=
      74 &&

    modelAgreement.agreement >=
      48 &&

    !severeConflict &&

    human.score <
      50
  ) {

    verdict =
      'Strong AI evidence';

  } else if (
    language ===
      'English' &&

    activeModels >=
      2 &&

    calibrated >=
      74 &&

    sufficiency.score >=
      62 &&

    modelAgreement.agreement >=
      38 &&

    modelAgreement.spread <
      55 &&

    human.score <
      55
  ) {

    verdict =
      'Likely AI';

  }


  /*
    HUMAN verdict requires more than simply low AI score.
  */

  if (
    language ===
      'English' &&

    activeModels >=
      2 &&

    calibrated <=
      18 &&

    human.score >=
      50 &&

    sufficiency.score >=
      60 &&

    modelAgreement.agreement >=
      45
  ) {

    verdict =
      'Strong human evidence';

  } else if (
    language ===
      'English' &&

    activeModels >=
      2 &&

    calibrated <=
      34 &&

    human.score >=
      40 &&

    sufficiency.score >=
      50 &&

    modelAgreement.spread <
      50
  ) {

    verdict =
      'Likely human';
  }


  /*
    Conflict defense.
  */

  if (
    highConflict &&
    (
      verdict.includes(
        'AI'
      ) ||
      verdict.includes(
        'human'
      )
    )
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  if (
    sufficiency.score <
    55
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  if (
    language !==
    'English'
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  return {

    raw,

    weightedRaw:
      Math.round(
        weightedRaw
      ),

    rawMedian:
      Math.round(
        rawMedian
      ),

    calibrated,

    uncertainty,

    confidence,

    verdict,

    activeModels,

    modelWeights,

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

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    thirdUsed
  };
}


/* ============================================================
   MAIN SMART SCAN
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
    words <
    80
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

    setProgress(
      3,
      'Profiling document…'
    );


    const language =
      detectLanguage(
        value
      );


    const profile =
      createProfile(
        value
      );


    const domainInfo =
      estimateDomain(
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


    let segmentScores = [];


    /*
      MODEL A
    */

    try {

      const model =
        await loadTMR();


      setProgress(
        24,
        'Running TMR detector…'
      );


      scores.tmr =
        await classify(
          model,
          value
        );


      for (
        let i = 0;
        i < chunks.length;
        i++
      ) {

        setProgress(
          28 +
          Math.round(
            (
              i /
              Math.max(
                1,
                chunks.length
              )
            ) *
            24
          ),
          `Trace Map ${i + 1}/${chunks.length}`
        );


        segmentScores.push(
          await classify(
            model,
            chunks[
              i
            ]
          )
        );
      }

    } catch (
      error
    ) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /*
      MODEL B
    */

    try {

      const model =
        await loadE5();


      setProgress(
        58,
        'Running E5-small…'
      );


      scores.e5 =
        await classify(
          model,
          value
        );

    } catch (
      error
    ) {

      console.error(
        'E5 failed:',
        error
      );
    }


    const mobile =
      isMobileDevice();


    let thirdUsed =
      false;


    if (
      !mobile &&
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

          words
        });
    }


    /*
      MODEL C
    */

    if (
      thirdUsed
    ) {

      try {

        const model =
          await loadModern();


        setProgress(
          74,
          'Running ModernBERT judge…'
        );


        scores.modern =
          await classify(
            model,
            value
          );

      } catch (
        error
      ) {

        console.error(
          'ModernBERT failed:',
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
          () =>
            50
        );
    }


    setProgress(
      90,
      'Adaptive calibration…'
    );


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

        benchmarkRecords
      });


    const reliability =
      consensus.reliability;


    const scan = {

      version:
        VERSION,

      timestamp:
        nowISO(),

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

      reliability,

      consensus
    };


    renderScan(
      scan
    );


    saveHistory(
      scan
    );


    setProgress(
      100,
      'Trace complete'
    );


    if (
      mobile
    ) {

      setState(
        'V6.3 Mobile Safe ✓'
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V6.3 Adaptive 3-model engine ✓'
      );

    } else {

      setState(
        'V6.3 Adaptive 2-model engine ✓'
      );
    }


    setTimeout(
      () => {

        benchmarkPrompt(
          scan
        );

      },
      700
    );

  } catch (
    error
  ) {

    console.error(
      'Fatal scan error:',
      error
    );


    setState(
      'Scan error'
    );


    alert(
      'AI Trace encountered an error during analysis. Refresh the page and try again.'
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
    consensus.confidence >=
      75
      ? 'High'
      : consensus.confidence >=
          50
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

    let text =
      `Diagnostic detector signal: ${consensus.calibrated}%. ` +
      `Evidence sufficiency: ${consensus.sufficiency.score}% (${consensus.sufficiency.level}). ` +
      `Model agreement: ${consensus.modelAgreement.agreement}%.`;


    if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {

      text =
        `AI Trace abstained because the available evidence was not strong enough for a reliable AI/Human attribution. ` +
        text;
    }


    if (
      consensus.outlier.detected
    ) {

      text +=
        ` ${String(
          consensus.outlier.detector
        ).toUpperCase()} was down-weighted as a possible detector outlier.`;
    }


    $('explain').textContent =
      text;
  }


  const humanDisplay =
    clamp(
      Math.round(
        human.score *
        0.72 +
        (
          100 -
          consensus.calibrated
        ) *
        0.28
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
        'V6.3 • ADAPTIVE OUTLIER DEFENSE';

    } else if (
      consensus.thirdUsed
    ) {

      $('engineBadge').textContent =
        'V6.3 • 3-MODEL CALIBRATED';

    } else {

      $('engineBadge').textContent =
        'V6.3 • ADAPTIVE CONSENSUS';
    }
  }


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


  const reliabilityLine =
    detector => {

      const reliability =
        consensus.reliability[
          detector
        ];

      if (
        !reliability
      ) {

        return 'No calibration data';
      }

      return (
        `Base ${Number(
          reliability.base
        ).toFixed(2)} · ` +
        `AI ${Number(
          reliability.ai.weight
        ).toFixed(2)} · ` +
        `Human ${Number(
          reliability.human.weight
        ).toFixed(2)}`
      );
    };


  const modelWeightLine =
    detector => {

      const item =
        consensus.modelWeights[
          detector
        ];

      if (
        !item
      ) {

        return 'Inactive';
      }

      return (
        `${item.score}% signal · ` +
        `effective weight ${item.weight}`
      );
    };


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
      'TMR detector',
      Number.isFinite(
        scores.tmr
      )
        ? `${scores.tmr}% AI signal`
        : 'Unavailable',
      'Model A'
    ],

    [
      'TMR adaptive weight',
      modelWeightLine(
        'tmr'
      ),
      reliabilityLine(
        'tmr'
      )
    ],

    [
      'E5-small detector',
      Number.isFinite(
        scores.e5
      )
        ? `${scores.e5}% AI signal`
        : 'Unavailable',
      'Model B'
    ],

    [
      'E5 adaptive weight',
      modelWeightLine(
        'e5'
      ),
      reliabilityLine(
        'e5'
      )
    ],

    [
      'ModernBERT judge',
      Number.isFinite(
        scores.modern
      )
        ? `${scores.modern}% AI signal`
        : 'Not required / unavailable',
      'Model C'
    ],

    [
      'Modern adaptive weight',
      modelWeightLine(
        'modern'
      ),
      reliabilityLine(
        'modern'
      )
    ],

    [
      'Human counter-evidence',
      `${human.score}% — ${humanReasons}`,
      human.score >=
        55
        ? 'Strong'
        : human.score >=
            35
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
      `Range ${consensus.segmentAnalysis.range} pts`
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      domain ===
        'books' ||
      domain ===
        'poetry'
        ? 'Protected'
        : 'Context'
    ],

    [
      'Language fit',
      language ===
        'English'
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

    <span>${escapeHTML(
      item[
        0
      ]
    )}</span>

    <span>${escapeHTML(
      item[
        2
      ]
    )}</span>

  </div>

  <small>${escapeHTML(
    item[
      1
    ]
  )}</small>

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
        .toFixed(
          2
        ),

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

  <span>${escapeHTML(
    key
  )}</span>

  <b>${escapeHTML(
    value
  )}</b>

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
              score >=
                70
                ? 'AI-supporting'
                : score <=
                    30
                  ? 'Human-supporting'
                  : 'Uncertain';


            return `

<div class="segment">

  <div class="segmentHead">

    <b>
      Segment ${index + 1}
    </b>

    <span>
      ${score}% TMR · ${label}
    </span>

  </div>

  <div class="segmentMeter">

    <i style="width:${clamp(
      score
    )}%"></i>

  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        320
      )
    )}${
      chunk.length >
      320
        ? '…'
        : ''
    }
  </p>

</div>

`;

          }
        )
        .join('');
  }


  $('report')
    ?.scrollIntoView({

      behavior:
        'smooth',

      block:
        'start'
    });


  renderDatasetManager();

  renderCalibrationLab();
}


/* ============================================================
   BENCHMARK PROMPT
============================================================ */

function benchmarkPrompt(
  scan
) {

  const answer =
    prompt(
`AI TRACE V6.3 BENCHMARK

Only label samples whose true origin you KNOW.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known mixture
UNKNOWN = unknown origin

Cancel / leave empty to skip.`
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
    ) ||
    '';


  const records =
    loadBench();


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
  ];


  const sameType =
    records.filter(
      record =>
        record.truth ===
        truth
    ).length;


  const id =
    `${prefix}-${String(
      sameType + 1
    ).padStart(
      3,
      '0'
    )}`;


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
  records =
    loadBench()
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
          record.truth ===
          'AI'
      ).length,

    human:
      records.filter(
        record =>
          record.truth ===
          'HUMAN'
      ).length,

    mixed:
      records.filter(
        record =>
          record.truth ===
          'MIXED'
      ).length,

    unknown:
      records.filter(
        record =>
          record.truth ===
          'UNKNOWN'
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
   DATASET MANAGER RENDER
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


  if (
    $('datasetTotal')
  ) {

    $('datasetTotal').textContent =
      summary.total;
  }


  if (
    $('datasetBinary')
  ) {

    $('datasetBinary').textContent =
      summary.binary;
  }


  if (
    $('datasetAI')
  ) {

    $('datasetAI').textContent =
      summary.ai;
  }


  if (
    $('datasetHuman')
  ) {

    $('datasetHuman').textContent =
      summary.human;
  }


  if (
    $('datasetMixed')
  ) {

    $('datasetMixed').textContent =
      summary.mixed;
  }


  if (
    $('datasetUnknown')
  ) {

    $('datasetUnknown').textContent =
      summary.unknown;
  }


  if (
    $('datasetDomains')
  ) {

    $('datasetDomains').textContent =
      summary.domains;
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

    <span>
      ${escapeHTML(
        record.id ||
        'Record'
      )}
    </span>

    <span>
      ${escapeHTML(
        record.truth ||
        '?'
      )}
    </span>

  </div>

  <small>
    ${escapeHTML(
      record.domain ||
      'general'
    )}
    · ${escapeHTML(
      record.source ||
      'No source'
    )}
    · ${wordCount(
      record.text ||
      ''
    ) || record.words || 0} words
  </small>

</div>

`
            )
            .join('')
        : `

<div class="ev">

  <small>
    No V6.3 dataset records yet.
  </small>

</div>

`;
  }
}


/* ============================================================
   DOMAIN PERFORMANCE
============================================================ */

function domainPerformance(
  records =
    loadBench()
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
          rows
        ]
      ) => ({

        domain,

        ...evaluatePredictions(
          rows,
          benchmarkPrediction
        )
      })
    );
}


/* ============================================================
   INSPECTORS
============================================================ */

function falsePositiveRecords(
  records =
    loadBench()
) {

  return binaryRecords(
    records
  )
    .filter(
      record =>
        record.truth ===
          'HUMAN' &&
        benchmarkPrediction(
          record
        ) ===
          'AI'
    );
}


function falseNegativeRecords(
  records =
    loadBench()
) {

  return binaryRecords(
    records
  )
    .filter(
      record =>
        record.truth ===
          'AI' &&
        benchmarkPrediction(
          record
        ) ===
          'HUMAN'
    );
}


function abstentionRecords(
  records =
    loadBench()
) {

  return binaryRecords(
    records
  )
    .filter(
      record =>
        benchmarkPrediction(
          record
        ) ===
          'ABSTAIN'
    );
}


/* ============================================================
   CALIBRATION LAB
============================================================ */

function renderCalibrationLab() {

  const records =
    loadBench();


  const rows =
    binaryRecords(
      records
    );


  const readiness =
    benchmarkReadiness(
      records
    );


  const ensemble =
    evaluatePredictions(
      rows,
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

    <span>
      ${readiness.level}
    </span>

    <span>
      ${
        readiness.level ===
        'COLLECTING'
          ? 'LEARNING LIMITED'
          : 'CALIBRATION ACTIVE'
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
              label,
              value
            ]
          ) => `

<div class="metric">

  <span>
    ${escapeHTML(
      label
    )}
  </span>

  <b>
    ${escapeHTML(
      value
    )}
  </b>

</div>

`
        )
        .join('');
  }


  /*
    Detector reliability
  */

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

    <span>
      ${escapeHTML(
        detector.toUpperCase()
      )}
    </span>

    <span>
      ${item.global.samples} samples
    </span>

  </div>

  <small>
    Reliability ${Number(
      item.base
    ).toFixed(
      2
    )}
    · AI weight ${Number(
      item.ai.weight
    ).toFixed(
      2
    )}
    · Human weight ${Number(
      item.human.weight
    ).toFixed(
      2
    )}
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


  /*
    Domain performance
  */

  if (
    $('domainPerformance')
  ) {

    const domainRows =
      domainPerformance(
        records
      );


    $('domainPerformance').innerHTML =
      domainRows.length
        ? domainRows
            .map(
              item => `

<div class="metric">

  <span>
    ${escapeHTML(
      item.domain
    )}
  </span>

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


function renderInspector(
  elementId,
  records,
  emptyText
) {

  const element =
    $(
      elementId
    );


  if (
    !element
  ) {

    return;
  }


  if (
    !records.length
  ) {

    element.innerHTML = `

<div class="ev">

  <small>
    ${escapeHTML(
      emptyText
    )}
  </small>

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

    <span>
      ${escapeHTML(
        record.id ||
        'Record'
      )}
    </span>

    <span>
      ${escapeHTML(
        record.truth ||
        '?'
      )}
    </span>

  </div>

  <small>
    Domain ${escapeHTML(
      record.domain ||
      'general'
    )}
    · Prediction ${escapeHTML(
      benchmarkPrediction(
        record
      )
    )}
    · Signal ${record.consensus?.calibrated ?? '?'}%
    · Sufficiency ${record.consensus?.sufficiency?.score ?? '?'}%
  </small>

</div>

`
      )
      .join('');
}


/* ============================================================
   BULK IMPORT
============================================================ */

function parseBulkImport(
  raw
) {

  const lines =
    raw
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
          ) <
          30
        ) {

          throw new Error(
            'Sample text is too short'
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
              'general'
            ),

          text:
            sampleText
        });

      } catch (
        error
      ) {

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
          <b>${parsed.samples.length} valid samples.</b><br>
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
    loadJSON(
      IMPORT_KEY,
      []
    );


  const imported =
    parsed.samples.map(
      sample => ({

        importId:
          `IMP-${Date.now()}-${Math.random()
            .toString(
              36
            )
            .slice(
              2,
              8
            )}`,

        status:
          'PENDING',

        createdAt:
          nowISO(),

        ...sample
      })
    );


  saveJSON(
    IMPORT_KEY,
    [
      ...queue,
      ...imported
    ]
  );


  if (
    $('bulkImportResult')
  ) {

    $('bulkImportResult').innerHTML =
      `<b>${imported.length} samples added to the local calibration queue.</b><br>
       Bulk model execution will be added in the next calibration worker build.`;
  }


  alert(
    `${imported.length} samples imported into the calibration queue.`
  );
}


/* ============================================================
   EXPORT JSON
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

    domainPerformance:
      domainPerformance(
        records
      ),

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
    `AI-Trace-V63-Benchmark-${Date.now()}.json`;


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
   EXPORT CSV
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
      headers.join(
        ','
      ),
      ...rows
    ]
      .join(
        '\n'
      );


  const blob =
    new Blob(
      [
        csv
      ],
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
    `AI-Trace-V63-Benchmark-${Date.now()}.csv`;


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

if (
  textEl
) {

  textEl.addEventListener(
    'input',
    updateCount
  );
}


$('scan')
  ?.addEventListener(
    'click',
    runSmartScan
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
          'Delete all V6.3 benchmark records from this browser?'
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
        'V6.3 benchmark dataset deleted.'
      );
    }
  );


/* ============================================================
   DEVELOPER API
============================================================ */

window.AITraceV63 = {

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

      falsePositives:
        falsePositiveRecords(
          records
        ),

      falseNegatives:
        falseNegativeRecords(
          records
        ),

      abstentions:
        abstentionRecords(
          records
        ),

      records
    };
  },


  reliability(
    domain =
      'general'
  ) {

    return buildModelReliability(
      domain,
      loadBench()
    );
  },


  importedQueue() {

    return loadJSON(
      IMPORT_KEY,
      []
    );
  },


  history() {

    return loadJSON(
      HISTORY_KEY,
      []
    );
  },


  exportJSON() {

    exportBenchmarkJSON();
  },


  exportCSV() {

    exportBenchmarkCSV();
  },


  clearBenchmark() {

    localStorage.removeItem(
      BENCH_KEY
    );

    renderDatasetManager();

    renderCalibrationLab();
  },


  clearImports() {

    localStorage.removeItem(
      IMPORT_KEY
    );
  },


  clearHistory() {

    localStorage.removeItem(
      HISTORY_KEY
    );
  }
};


/* ============================================================
   INITIALIZE
============================================================ */

updateCount();

renderDatasetManager();

renderCalibrationLab();


console.info(
  `AI TRACE V${VERSION} loaded successfully`
);
