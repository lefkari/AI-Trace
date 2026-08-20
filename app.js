/*
  AI TRACE V6 — SMART CONSENSUS + RELIABILITY LAB
  PART 1 / 3

  Core:
  - TMR detector
  - E5-small detector
  - ModernBERT conditional judge
  - Human counter-evidence
  - Domain routing
  - Segment analysis
  - Outlier defense
  - Reliability weighting
  - Abstention logic
  - Benchmark-aware calibration
  - Zero paid API
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const VERSION = '6.1';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCH_KEY =
  'aiTraceBenchmarkV6';

const HISTORY_KEY =
  'aiTraceHistoryV6';

let tmrModel = null;
let e5Model = null;
let modernModel = null;

const $ = id =>
  document.getElementById(id);

const textEl =
  $('text');


/* =========================================================
   DEVICE
========================================================= */

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


/* =========================================================
   UI EVENTS
========================================================= */

if (textEl) {
  textEl.addEventListener(
    'input',
    updateCount
  );
}

if ($('clear')) {
  $('clear').addEventListener(
    'click',
    () => {
      textEl.value = '';
      updateCount();

      $('report')
        ?.classList
        .add('hidden');
    }
  );
}

if ($('demo')) {
  $('demo').addEventListener(
    'click',
    loadDemo
  );
}

if ($('scan')) {
  $('scan').addEventListener(
    'click',
    runSmartScan
  );
}


/* =========================================================
   BASIC UI
========================================================= */

function updateCount() {
  const words =
    wordCount(
      textEl?.value || ''
    );

  if ($('count')) {
    $('count').textContent =
      `${words} words`;
  }
}

function loadDemo() {
  if (!textEl) return;

  textEl.value = `Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

These capabilities create important opportunities for businesses, researchers, educators, and individuals. AI systems can process large amounts of information quickly and can sometimes identify useful patterns that would be difficult for people to notice manually.

However, rapid progress also creates new challenges. Artificially generated content can be persuasive, inaccurate, misleading, or difficult to distinguish from material created directly by a person. This makes transparency increasingly important.

Reliable AI detection will therefore depend on careful evaluation, transparent limitations, and continuous testing across many different types of content. A responsible detection system should not rely on one score alone, but instead combine multiple signals, measure disagreement, recognize uncertainty, and avoid making strong claims when the available evidence is weak.`;

  updateCount();
}

function setProgress(
  percent,
  label
) {
  const progressEl =
    $('progress');

  if (progressEl) {
    progressEl.classList.remove(
      'hidden'
    );
  }

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
        .add('hidden');
    },
    500
  );
}

function setState(label) {
  if ($('modelState')) {
    $('modelState').textContent =
      label;
  }
}


/* =========================================================
   HELPERS
========================================================= */

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

function average(values) {
  const usable =
    values.filter(
      Number.isFinite
    );

  if (!usable.length) {
    return 0;
  }

  return (
    usable.reduce(
      (sum, value) =>
        sum + value,
      0
    ) /
    usable.length
  );
}

function weightedAverage(
  values,
  weights
) {
  let numerator = 0;
  let denominator = 0;

  for (
    let i = 0;
    i < values.length;
    i++
  ) {
    const value =
      values[i];

    const weight =
      weights[i];

    if (
      Number.isFinite(value) &&
      Number.isFinite(weight) &&
      weight > 0
    ) {
      numerator +=
        value * weight;

      denominator +=
        weight;
    }
  }

  if (!denominator) {
    return 50;
  }

  return numerator /
    denominator;
}

function standardDeviation(
  values
) {
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

function median(values) {
  const usable =
    values
      .filter(
        Number.isFinite
      )
      .sort(
        (a, b) =>
          a - b
      );

  if (!usable.length) {
    return 50;
  }

  const middle =
    Math.floor(
      usable.length /
      2
    );

  if (
    usable.length % 2
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

function wordCount(value) {
  if (
    !value ||
    !value.trim()
  ) {
    return 0;
  }

  return value
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .length;
}

function escapeHTML(value) {
  return String(value).replace(
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

function detectLanguage(value) {
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
    latin / total >
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
    value.match(regex) ||
    []
  ).length;
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

  const avgSentence =
    average(
      sentenceLengths
    );

  const sentenceSD =
    standardDeviation(
      sentenceLengths
    );

  const punctuationPatterns = [
    /,/g,
    /;/g,
    /:/g,
    /[—–-]/g,
    /["“”‘’']/g,
    /[()]/g
  ];

  const punctuationTypes =
    punctuationPatterns
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
    (
      value.match(
        /["“”‘’]/g
      ) || []
    ).length;

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

  const firstPerson =
    (
      value.match(
        /\b(I|me|my|mine|we|us|our|ours)\b/gi
      ) || []
    ).length;

  const subjectiveMarkers =
    (
      value.match(
        /\b(I think|I believe|I suppose|I remember|I feel|in my view|it seems to me|perhaps|maybe|I do not know)\b/gi
      ) || []
    ).length;

  const contractions =
    (
      value.match(
        /\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi
      ) || []
    ).length;

  const transitions =
    (
      value.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result|on the other hand)\b/gi
      ) || []
    ).length;

  const openerTokens =
    sentences
      .map(
        sentence =>
          sentence
            .trim()
            .split(/\s+/)
            .slice(0, 2)
            .join(' ')
            .toLowerCase()
            .replace(
              /[^\p{L}\s]/gu,
              ''
            )
      )
      .filter(Boolean);

  const openerDiversity =
    new Set(
      openerTokens
    ).size /
    Math.max(
      1,
      openerTokens.length
    );

  return {
    words:
      words.length,

    sentences:
      sentences.length,

    paragraphs:
      paragraphs.length,

    lineBreaks:
      Math.max(
        0,
        lines.length - 1
      ),

    averageLineLength:
      lines.length
        ? value.length /
          lines.length
        : value.length,

    averageSentenceLength:
      avgSentence,

    sentenceDeviation:
      sentenceSD,

    sentenceBurstiness:
      avgSentence
        ? sentenceSD /
          avgSentence
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

    dialogueLines,

    firstPerson,

    subjectiveMarkers,

    contractions,

    transitions,

    openerDiversity
  };
}


/* =========================================================
   DOMAIN ESTIMATION
========================================================= */

function estimateDomain(
  value,
  profile
) {
  const content =
    value.toLowerCase();

  const signals = {
    abstracts:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract)\b/g
      ),

    recipes:
      countMatches(
        content,
        /\b(cup|tablespoon|teaspoon|ingredients|preheat|oven|bake|stir|chop|minutes|serve)\b/g
      ),

    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|experience)\b/g
      ),

    reddit:
      countMatches(
        content,
        /\b(aita|tldr|subreddit|upvote|downvote|edit:|throwaway|imo|lol)\b/g
      ),

    wiki:
      countMatches(
        content,
        /\b(was born|is a|refers to|located in|population|history of|known for|founded|species)\b/g
      ),

    news:
      countMatches(
        content,
        /\b(reuters|reported|according to|officials|government|minister|president|announced|agency)\b/g
      ),

    poetry:
      (
        profile.lineBreaks >= 6 &&
        profile.averageLineLength < 60
      )
        ? 4
        : 0,

    books:
      (
        profile.quoteCount >= 6 ||
        profile.dialogueLines >= 2
      )
        ? 4
        : 0
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
  ] = sorted[0];

  if (
    !score ||
    score < 2
  ) {
    if (
      profile.quoteCount >= 4 ||
      profile.dialogueLines > 0
    ) {
      return {
        domain:
          'books',

        confidence:
          'low'
      };
    }

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


/* =========================================================
   HUMAN COUNTER-EVIDENCE
========================================================= */

function humanEvidence(
  profile,
  domain
) {
  let score = 0;

  const reasons = [];

  if (
    profile.sentenceBurstiness >=
    0.75
  ) {
    score += 24;

    reasons.push(
      'high sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.50
  ) {
    score += 15;

    reasons.push(
      'moderate sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.35
  ) {
    score += 7;
  }

  if (
    profile.punctuationTypes >=
    5
  ) {
    score += 14;

    reasons.push(
      'rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >=
    3
  ) {
    score += 7;
  }

  if (
    profile.quoteCount >= 8 ||
    profile.dialogueLines >= 2
  ) {
    score += 16;

    reasons.push(
      'dialogue or quotation structure'
    );

  } else if (
    profile.quoteCount >= 3
  ) {
    score += 7;
  }

  if (
    profile.firstPerson >= 4 ||
    profile.subjectiveMarkers >= 2
  ) {
    score += 12;

    reasons.push(
      'personal or subjective voice'
    );

  } else if (
    profile.firstPerson > 0 ||
    profile.subjectiveMarkers > 0
  ) {
    score += 5;
  }

  if (
    profile.contractions >= 4
  ) {
    score += 6;

    reasons.push(
      'natural contraction usage'
    );
  }

  if (
    profile.paragraphDeviation >= 25 &&
    profile.paragraphs >= 3
  ) {
    score += 9;

    reasons.push(
      'irregular paragraph rhythm'
    );
  }

  if (
    profile.lexicalDiversity >=
    0.66
  ) {
    score += 8;

  } else if (
    profile.lexicalDiversity >=
    0.55
  ) {
    score += 4;
  }

  if (
    profile.openerDiversity >=
    0.85
  ) {
    score += 8;

    reasons.push(
      'diverse sentence openings'
    );
  }

  if (
    profile.transitions >= 5
  ) {
    score -= 8;

  } else if (
    profile.transitions >= 3
  ) {
    score -= 4;
  }

  if (
    domain === 'books'
  ) {
    score += 16;

    reasons.push(
      'literary prose characteristics'
    );

    reasons.push(
      'literary-domain protection'
    );
  }

  if (
    domain === 'poetry'
  ) {
    score += 20;

    reasons.push(
      'poetic structure'
    );

    reasons.push(
      'literary-domain protection'
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


/* =========================================================
   CHUNKING
========================================================= */

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


/* =========================================================
   MODEL LOADERS
========================================================= */

async function loadTMR() {
  if (tmrModel) {
    return tmrModel;
  }

  setState(
    'Loading TMR…'
  );

  setProgress(
    8,
    'Loading Model A…'
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
  if (e5Model) {
    return e5Model;
  }

  setState(
    'Loading E5-small…'
  );

  setProgress(
    18,
    'Loading Model B…'
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
  if (modernModel) {
    return modernModel;
  }

  setState(
    'Loading ModernBERT judge…'
  );

  setProgress(
    72,
    'Deep verification: loading Model C…'
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


/* =========================================================
   MODEL OUTPUT NORMALIZATION
========================================================= */

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
        item.label || ''
      ).toLowerCase();

    const score =
      Number(
        item.score
      ) || 0;

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
  const result =
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
      result
    ) * 100
  );
}


/* =========================================================
   BENCHMARK STORAGE
========================================================= */

function loadBench() {
  try {
    return JSON.parse(
      localStorage.getItem(
        BENCH_KEY
      ) || '[]'
    );

  } catch {
    return [];
  }
}

function saveBench(records) {
  try {
    localStorage.setItem(
      BENCH_KEY,
      JSON.stringify(
        records
      )
    );

  } catch (error) {
    console.warn(
      'Benchmark save failed:',
      error
    );
  }
}


/* =========================================================
   PREDICTION NORMALIZATION
========================================================= */

function prediction(record) {
  const verdict =
    record?.consensus?.verdict ||
    record?.verdict ||
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


/* =========================================================
   DETECTOR BENCHMARK HELPERS
========================================================= */

function detectorPrediction(
  score
) {
  if (
    !Number.isFinite(
      score
    )
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

function binaryRecords() {
  return loadBench()
    .filter(
      record =>
        record.truth === 'AI' ||
        record.truth === 'HUMAN'
    );
}

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
    const record
    of rows
  ) {
    const predicted =
      getPrediction(
        record
      );

    if (
      predicted ===
      'ABSTAIN'
    ) {
      if (
        record.truth ===
        'AI'
      ) {
        aiAbstain++;
      } else {
        humanAbstain++;
      }

      continue;
    }

    if (
      record.truth ===
        'AI' &&
      predicted ===
        'AI'
    ) {
      TP++;
    }

    if (
      record.truth ===
        'HUMAN' &&
      predicted ===
        'HUMAN'
    ) {
      TN++;
    }

    if (
      record.truth ===
        'HUMAN' &&
      predicted ===
        'AI'
    ) {
      FP++;
    }

    if (
      record.truth ===
        'AI' &&
      predicted ===
        'HUMAN'
    ) {
      FN++;
    }
  }

  const totalAI =
    rows.filter(
      record =>
        record.truth ===
        'AI'
    ).length;

  const totalHuman =
    rows.filter(
      record =>
        record.truth ===
        'HUMAN'
    ).length;

  const decided =
    TP +
    TN +
    FP +
    FN;

  const total =
    rows.length;

  const abstentions =
    aiAbstain +
    humanAbstain;

  return {
    total,
    totalAI,
    totalHuman,

    decided,
    abstentions,

    TP,
    TN,
    FP,
    FN,

    aiAbstain,
    humanAbstain,

    coverage:
      total
        ? Math.round(
            decided /
            total *
            100
          )
        : 0,

    selectiveAccuracy:
      decided
        ? Math.round(
            (
              TP +
              TN
            ) /
            decided *
            100
          )
        : 0,

    precision:
      TP + FP
        ? Math.round(
            TP /
            (
              TP +
              FP
            ) *
            100
          )
        : 0,

    recall:
      totalAI
        ? Math.round(
            TP /
            totalAI *
            100
          )
        : 0,

    specificity:
      totalHuman
        ? Math.round(
            TN /
            totalHuman *
            100
          )
        : 0,

    fpr:
      totalHuman
        ? Math.round(
            FP /
            totalHuman *
            100
          )
        : 0,

    fnr:
      totalAI
        ? Math.round(
            FN /
            totalAI *
            100
          )
        : 0,

    aiAbstainRate:
      totalAI
        ? Math.round(
            aiAbstain /
            totalAI *
            100
          )
        : 0,

    humanAbstainRate:
      totalHuman
        ? Math.round(
            humanAbstain /
            totalHuman *
            100
          )
        : 0
  };
}
/* =========================================================
   DETECTOR RELIABILITY
========================================================= */

function detectorReliability() {
  const rows =
    binaryRecords();

  const detectorNames = [
    'tmr',
    'e5',
    'modern'
  ];

  const result = {};

  for (
    const detector
    of detectorNames
  ) {
    const usable =
      rows.filter(
        record =>
          Number.isFinite(
            record?.scores?.[
              detector
            ]
          )
      );

    const metrics =
      evaluatePredictions(
        usable,
        record =>
          detectorPrediction(
            record.scores[
              detector
            ]
          )
      );

    let reliability = 1;

    if (
      usable.length >= 6
    ) {
      const accuracyFactor =
        metrics.selectiveAccuracy /
        100;

      const coverageFactor =
        metrics.coverage /
        100;

      const fpPenalty =
        1 -
        (
          metrics.fpr /
          100
        ) * 0.75;

      const fnPenalty =
        1 -
        (
          metrics.fnr /
          100
        ) * 0.55;

      reliability =
        (
          accuracyFactor *
          0.50 +
          coverageFactor *
          0.20 +
          fpPenalty *
          0.18 +
          fnPenalty *
          0.12
        );
    }

    reliability =
      clamp(
        reliability,
        0.35,
        1.35
      );

    result[
      detector
    ] = {
      samples:
        usable.length,

      metrics,

      weight:
        Number(
          reliability.toFixed(
            2
          )
        )
    };
  }

  return result;
}


/* =========================================================
   DOMAIN PERFORMANCE
========================================================= */

function domainPerformance() {
  const records =
    binaryRecords();

  const map =
    new Map();

  for (
    const record
    of records
  ) {
    const domain =
      record.domain ||
      'general';

    if (
      !map.has(
        domain
      )
    ) {
      map.set(
        domain,
        []
      );
    }

    map.get(
      domain
    ).push(
      record
    );
  }

  const result = {};

  for (
    const [
      domain,
      rows
    ]
    of map.entries()
  ) {
    result[
      domain
    ] =
      evaluatePredictions(
        rows,
        prediction
      );
  }

  return result;
}


/* =========================================================
   CALIBRATION READINESS
========================================================= */

function calibrationReadiness() {
  const rows =
    binaryRecords();

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

  const ready =
    (
      rows.length >= 20 &&
      ai >= 8 &&
      human >= 8 &&
      domains >= 2
    );

  return {
    ready,

    total:
      rows.length,

    ai,

    human,

    domains,

    message:
      ready
        ? 'READY'
        : 'NOT READY'
  };
}


/* =========================================================
   BENCHMARK-AWARE THRESHOLD SEARCH
========================================================= */

function searchCalibration() {
  const readiness =
    calibrationReadiness();

  if (
    !readiness.ready
  ) {
    return {
      ready:
        false,

      readiness
    };
  }

  const rows =
    binaryRecords();

  let best = null;

  for (
    let aiThreshold = 72;
    aiThreshold <= 94;
    aiThreshold += 2
  ) {
    for (
      let humanThreshold = 12;
      humanThreshold <= 38;
      humanThreshold += 2
    ) {
      for (
        let minQuality = 45;
        minQuality <= 80;
        minQuality += 5
      ) {
        let TP = 0;
        let TN = 0;
        let FP = 0;
        let FN = 0;
        let abstain = 0;

        for (
          const record
          of rows
        ) {
          const consensus =
            record.consensus || {};

          const score =
            consensus.calibrated ??
            consensus.weighted ??
            consensus.raw ??
            50;

          const quality =
            consensus.quality ??
            50;

          let predicted =
            'ABSTAIN';

          if (
            quality >=
            minQuality
          ) {
            if (
              score >=
              aiThreshold
            ) {
              predicted =
                'AI';

            } else if (
              score <=
              humanThreshold
            ) {
              predicted =
                'HUMAN';
            }
          }

          if (
            predicted ===
            'ABSTAIN'
          ) {
            abstain++;

            continue;
          }

          if (
            record.truth ===
              'AI' &&
            predicted ===
              'AI'
          ) {
            TP++;
          }

          if (
            record.truth ===
              'HUMAN' &&
            predicted ===
              'HUMAN'
          ) {
            TN++;
          }

          if (
            record.truth ===
              'HUMAN' &&
            predicted ===
              'AI'
          ) {
            FP++;
          }

          if (
            record.truth ===
              'AI' &&
            predicted ===
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

        const coverage =
          rows.length
            ? decided /
              rows.length
            : 0;

        const accuracy =
          decided
            ? (
                TP +
                TN
              ) /
              decided
            : 0;

        const objective =
          accuracy * 100 +
          coverage * 24 -
          FP * 22 -
          FN * 12 -
          abstain * 0.7;

        if (
          !best ||
          objective >
            best.objective
        ) {
          best = {
            ready:
              true,

            aiThreshold,

            humanThreshold,

            minQuality,

            TP,
            TN,
            FP,
            FN,

            abstain,

            coverage:
              Math.round(
                coverage * 100
              ),

            selectiveAccuracy:
              Math.round(
                accuracy * 100
              ),

            objective:
              Number(
                objective.toFixed(
                  1
                )
              )
          };
        }
      }
    }
  }

  return best;
}


/* =========================================================
   MODEL AGREEMENT
========================================================= */

function modelAgreementInfo(
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

  const aiModels =
    entries.filter(
      (
        [
          ,
          value
        ]
      ) =>
        value >= 70
    );

  const humanModels =
    entries.filter(
      (
        [
          ,
          value
        ]
      ) =>
        value <= 30
    );

  return {
    active:
      entries.length,

    aiModels:
      aiModels.length,

    humanModels:
      humanModels.length,

    aiStrong:
      aiModels.length >= 2,

    humanStrong:
      humanModels.length >= 2
  };
}


/* =========================================================
   OUTLIER DETECTION
========================================================= */

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

      value:
        null,

      pairGap:
        null,

      outlierDistance:
        null,

      pair:
        []
    };
  }

  const candidates = [];

  for (
    let i = 0;
    i <
    entries.length;
    i++
  ) {
    const [
      name,
      value
    ] =
      entries[i];

    const others =
      entries.filter(
        (
          ,
          index
        ) =>
          index !== i
      );

    const pairGap =
      Math.abs(
        others[0][1] -
        others[1][1]
      );

    const pairMean =
      average(
        others.map(
          item =>
            item[1]
        )
      );

    const outlierDistance =
      Math.abs(
        value -
        pairMean
      );

    candidates.push({
      detector:
        name,

      value,

      pairGap,

      outlierDistance,

      pair:
        others.map(
          item =>
            item[0]
        ),

      pairValues:
        others.map(
          item =>
            item[1]
        )
    });
  }

  candidates.sort(
    (
      a,
      b
    ) =>
      b.outlierDistance -
      a.outlierDistance
  );

  const best =
    candidates[0];

  const detected =
    (
      best.outlierDistance >=
        28 &&
      best.pairGap <=
        15
    );

  return {
    ...best,

    detected
  };
}


/* =========================================================
   THIRD MODEL ROUTING
========================================================= */

function shouldUseThirdModel({
  quickScores,
  segmentScores,
  domain,
  humanScore,
  words
}) {
  if (
    !Number.isFinite(
      quickScores.tmr
    ) ||
    !Number.isFinite(
      quickScores.e5
    )
  ) {
    return false;
  }

  const modelGap =
    Math.abs(
      quickScores.tmr -
      quickScores.e5
    );

  const raw =
    median(
      [
        quickScores.tmr,
        quickScores.e5
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

  const segmentSD =
    standardDeviation(
      segmentScores
    );

  return (
    words < 180 ||
    modelGap >= 18 ||
    (
      raw >= 35 &&
      raw <= 85
    ) ||
    segmentRange >= 45 ||
    segmentSD >= 20 ||
    humanScore >= 45 ||
    domain === 'books' ||
    domain === 'poetry'
  );
}


/* =========================================================
   RELIABILITY WEIGHTED SIGNAL
========================================================= */

function reliabilityWeightedSignal(
  scores
) {
  const reliability =
    detectorReliability();

  const values = [];
  const weights = [];

  if (
    Number.isFinite(
      scores.tmr
    )
  ) {
    values.push(
      scores.tmr
    );

    weights.push(
      reliability.tmr
        ?.weight ??
      1
    );
  }

  if (
    Number.isFinite(
      scores.e5
    )
  ) {
    values.push(
      scores.e5
    );

    weights.push(
      reliability.e5
        ?.weight ??
      1
    );
  }

  if (
    Number.isFinite(
      scores.modern
    )
  ) {
    values.push(
      scores.modern
    );

    weights.push(
      reliability.modern
        ?.weight ??
      1
    );
  }

  return Math.round(
    weightedAverage(
      values,
      weights
    )
  );
}


/* =========================================================
   SEGMENT CONFIRMATION
========================================================= */

function segmentConfirmation(
  segmentScores
) {
  const usable =
    segmentScores.filter(
      Number.isFinite
    );

  if (
    !usable.length
  ) {
    return {
      mean:
        50,

      range:
        100,

      sd:
        100,

      strongAI:
        false,

      strongHuman:
        false
    };
  }

  const mean =
    Math.round(
      average(
        usable
      )
    );

  const range =
    Math.round(
      Math.max(
        ...usable
      ) -
      Math.min(
        ...usable
      )
    );

  const sd =
    Math.round(
      standardDeviation(
        usable
      )
    );

  return {
    mean,
    range,
    sd,

    strongAI:
      mean >= 85 &&
      range <= 20,

    strongHuman:
      mean <= 25 &&
      range <= 20
  };
}


/* =========================================================
   CONSENSUS ENGINE
========================================================= */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  domainConfidence,
  human,
  thirdUsed
}) {
  const activeValues =
    [
      scores.tmr,
      scores.e5,
      scores.modern
    ].filter(
      Number.isFinite
    );

  const agreement =
    modelAgreementInfo(
      scores
    );

  const outlier =
    detectModelOutlier(
      scores
    );

  const segments =
    segmentConfirmation(
      segmentScores
    );

  const raw =
    Math.round(
      median(
        activeValues
      )
    );

  const weighted =
    reliabilityWeightedSignal(
      scores
    );

  let usableForConflict =
    activeValues.slice();

  if (
    outlier.detected
  ) {
    usableForConflict =
      outlier.pairValues;
  }

  const rawSpread =
    activeValues.length >
    1
      ? Math.max(
          ...activeValues
        ) -
        Math.min(
          ...activeValues
        )
      : 100;

  const effectiveConflict =
    usableForConflict.length >
    1
      ? Math.max(
          ...usableForConflict
        ) -
        Math.min(
          ...usableForConflict
        )
      : 100;

  const modelSD =
    Math.round(
      standardDeviation(
        usableForConflict
      )
    );

  let quality =
    100;

  quality -=
    Math.min(
      35,
      effectiveConflict *
      0.65
    );

  quality -=
    Math.min(
      25,
      segments.sd *
        0.70 +
      segments.range *
        0.15
    );

  if (
    profile.words < 120
  ) {
    quality -= 20;

  } else if (
    profile.words < 180
  ) {
    quality -= 12;

  } else if (
    profile.words < 250
  ) {
    quality -= 5;
  }

  if (
    language !==
    'English'
  ) {
    quality -= 35;
  }

  if (
    (
      domain ===
        'books' ||
      domain ===
        'poetry'
    ) &&
    domainConfidence ===
      'high'
  ) {
    quality -= 6;
  }

  if (
    activeValues.length ===
    1
  ) {
    quality -= 30;

  } else if (
    activeValues.length ===
    2
  ) {
    quality -= 8;
  }

  if (
    outlier.detected &&
    agreement.aiStrong
  ) {
    quality += 8;
  }

  if (
    outlier.detected &&
    agreement.humanStrong
  ) {
    quality += 8;
  }

  if (
    segments.strongAI &&
    agreement.aiModels >= 1
  ) {
    quality += 6;
  }

  if (
    segments.strongHuman &&
    agreement.humanModels >= 1
  ) {
    quality += 6;
  }

  quality =
    clamp(
      Math.round(
        quality
      )
    );

  const instability =
    1 -
    quality /
    100;

  const humanPenalty =
    human.score *
    (
      0.08 +
      instability *
      0.55
    ) *
    (
      weighted /
      100
    );

  let calibrated =
    Math.round(
      weighted -
      humanPenalty
    );

  if (
    outlier.detected &&
    agreement.aiStrong
  ) {
    calibrated =
      Math.round(
        average(
          outlier.pairValues
        )
      );
  }

  if (
    outlier.detected &&
    agreement.humanStrong
  ) {
    calibrated =
      Math.round(
        average(
          outlier.pairValues
        )
      );
  }

  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    human.score >= 55 &&
    !agreement.aiStrong
  ) {
    calibrated =
      Math.min(
        calibrated,
        64
      );
  }

  calibrated =
    clamp(
      calibrated
    );

  const calibration =
    searchCalibration();

  let aiThreshold =
    86;

  let humanThreshold =
    28;

  let minQuality =
    60;

  if (
    calibration.ready
  ) {
    aiThreshold =
      calibration.aiThreshold;

    humanThreshold =
      calibration.humanThreshold;

    minQuality =
      calibration.minQuality;
  }

  const severeConflict =
    (
      effectiveConflict >= 38 ||
      segments.range >= 75 ||
      segments.sd >= 30
    );

  const unresolvedModelConflict =
    (
      rawSpread >= 45 &&
      !outlier.detected &&
      !agreement.aiStrong &&
      !agreement.humanStrong
    );

  const literaryRisk =
    (
      (
        domain === 'books' ||
        domain === 'poetry'
      ) &&
      domainConfidence ===
        'high' &&
      human.score >= 50
    );

  let verdict =
    'INCONCLUSIVE';

  if (
    language !==
    'English'
  ) {
    verdict =
      'INCONCLUSIVE';

  } else if (
    severeConflict ||
    unresolvedModelConflict
  ) {
    verdict =
      'CONFLICTING EVIDENCE';

  } else if (
    calibrated >=
      aiThreshold &&
    quality >=
      minQuality &&
    human.score < 55 &&
    (
      agreement.aiStrong ||
      segments.strongAI
    )
  ) {
    verdict =
      calibrated >= 86
        ? 'Strong AI evidence'
        : 'Likely AI';

  } else if (
    calibrated <=
      humanThreshold &&
    quality >=
      minQuality &&
    human.score >= 45 &&
    (
      agreement.humanStrong ||
      segments.strongHuman
    )
  ) {
    verdict =
      calibrated <= 18
        ? 'Strong human evidence'
        : 'Likely human';

  } else {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    literaryRisk &&
    verdict.includes(
      'AI'
    ) &&
    !agreement.aiStrong
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  const uncertainty =
    clamp(
      Math.round(
        100 -
        quality +
        (
          verdict ===
          'CONFLICTING EVIDENCE'
            ? 20
            : 0
        )
      ),
      5,
      95
    );

  const confidence =
    100 -
    uncertainty;

  return {
    raw,

    weighted,

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

    rawSpread:
      Math.round(
        rawSpread
      ),

    effectiveConflict:
      Math.round(
        effectiveConflict
      ),

    modelSD,

    segmentRange:
      segments.range,

    segmentSD:
      segments.sd,

    segmentMean:
      segments.mean,

    strongSegmentAI:
      segments.strongAI,

    strongSegmentHuman:
      segments.strongHuman,

    activeModels:
      activeValues.length,

    thirdUsed,

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    agreement,

    outlier,

    calibrationReady:
      calibration.ready,

    aiThreshold,

    humanThreshold,

    minQuality
  };
}


/* =========================================================
   MAIN SMART SCAN
========================================================= */

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
    !$('scan')
  ) {
    return;
  }

  $('scan').disabled =
    true;

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

    let segmentScores =
      [];

    let thirdUsed =
      false;

    let mobile =
      isMobileDevice();

    /* MODEL A */

    try {
      const modelA =
        await loadTMR();

      setProgress(
        22,
        'Running Model A…'
      );

      scores.tmr =
        await classify(
          modelA,
          value
        );

      for (
        let i = 0;
        i <
        chunks.length;
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
            28
          ),

          `Trace Map ${i + 1}/${chunks.length}…`
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
        'TMR failed:',
        error
      );
    }

    /* MODEL B */

    try {
      const modelB =
        await loadE5();

      setProgress(
        60,
        'Running Model B…'
      );

      scores.e5 =
        await classify(
          modelB,
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

    if (
      Number.isFinite(
        scores.tmr
      ) &&
      Number.isFinite(
        scores.e5
      )
    ) {
      thirdUsed =
        !mobile &&
        shouldUseThirdModel({
          quickScores:
            scores,

          segmentScores,

          domain:
            domainInfo.domain,

          humanScore:
            human.score,

          words
        });
    }

    /* MODEL C */

    if (
      thirdUsed
    ) {
      try {
        const modelC =
          await loadModern();

        setProgress(
          76,
          'Running Model C…'
        );

        scores.modern =
          await classify(
            modelC,
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
      'Calibrating evidence…'
    );

    const consensus =
      buildConsensus({
        scores,

        segmentScores,

        profile,

        language,

        domain:
          domainInfo.domain,

        domainConfidence:
          domainInfo.confidence,

        human,

        thirdUsed
      });

    const scan = {
      version:
        VERSION,

      timestamp:
        new Date()
          .toISOString(),

      words,

      language,

      domain:
        domainInfo.domain,

      domainConfidence:
        domainInfo.confidence,

      mobile,

      profile,

      scores,

      segmentScores,

      human,

      consensus
    };

    render(
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
        'V6 Mobile Smart Engine ✓'
      );

    } else if (
      thirdUsed
    ) {
      setState(
        'V6 Deep 3-model engine ✓'
      );

    } else {
      setState(
        'V6 Smart engine ✓'
      );
    }

    setTimeout(
      () => {
        try {
          benchmarkPrompt(
            scan
          );

        } catch (
          benchmarkError
        ) {
          console.warn(
            'Benchmark prompt error:',
            benchmarkError
          );
        }
      },
      700
    );

  } catch (
    fatalError
  ) {
    console.error(
      'Fatal scan error:',
      fatalError
    );

    setState(
      'Scan error'
    );

    alert(
      'AI Trace encountered an error during analysis. Refresh the page and try again.'
    );

  } finally {
    $('scan').disabled =
      false;

    hideProgress();
  }
}
/* =========================================================
   RENDER MAIN REPORT
========================================================= */

function render(scan) {
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
    .remove('hidden');

  /*
    V6.1:
    If evidence is insufficient or conflicting,
    do not visually present the detector signal
    as if it were a probability of authorship.
  */

  const resolvedDecision =
    consensus.verdict !==
      'INCONCLUSIVE' &&
    consensus.verdict !==
      'CONFLICTING EVIDENCE';

  if ($('score')) {
    $('score').textContent =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '—';
  }

  if ($('scaleFill')) {
    $('scaleFill').style.width =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '0%';
  }

  if ($('verdict')) {
    $('verdict').textContent =
      consensus.verdict;
  }

  const confidenceLabel =
    consensus.confidence >= 75
      ? 'High'
      : consensus.confidence >= 50
        ? 'Medium'
        : 'Low';

  if ($('confidence')) {
    $('confidence').textContent =
      `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;
  }

  if ($('explain')) {
    let explanation = '';

    if (
      consensus.verdict ===
      'CONFLICTING EVIDENCE'
    ) {
      explanation =
        `The detectors produced materially conflicting evidence. ` +
        `The diagnostic signal was ${consensus.calibrated}%, but AI Trace does not present that value as an authorship probability. ` +
        `Evidence quality was ${consensus.quality}% and effective model conflict was ${consensus.effectiveConflict} points.`;

    } else if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {
      explanation =
        `The available evidence is insufficient for a reliable AI/Human decision. ` +
        `Diagnostic detector signal: ${consensus.calibrated}%. ` +
        `Evidence quality: ${consensus.quality}%. ` +
        `Active independent models: ${consensus.activeModels}/3.`;

    } else {
      explanation =
        `The final decision combines detector consensus, reliability weighting, segment stability, human counter-evidence and domain context. ` +
        `Diagnostic detector signal: ${consensus.calibrated}%. ` +
        `Evidence quality: ${consensus.quality}%.`;
    }

    if (
      consensus.outlier?.detected
    ) {
      explanation +=
        ` ${String(
          consensus.outlier.detector
        ).toUpperCase()} was identified as a possible detector outlier.`;
    }

    if (
      consensus.calibrationReady
    ) {
      explanation +=
        ' Local benchmark calibration is active.';
    } else {
      explanation +=
        ' Benchmark calibration is still collecting data.';
    }

    $('explain').textContent =
      explanation;
  }


  /* =======================================================
     TRACE DNA
  ======================================================= */

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

  if ($('humanVal')) {
    $('humanVal').textContent =
      `${humanDisplay}%`;
  }

  if ($('aiVal')) {
    $('aiVal').textContent =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : 'N/A';
  }

  if ($('uncertainVal')) {
    $('uncertainVal').textContent =
      `${consensus.uncertainty}%`;
  }

  if ($('humanBar')) {
    $('humanBar').style.width =
      `${humanDisplay}%`;
  }

  if ($('aiBar')) {
    $('aiBar').style.width =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '0%';
  }

  if ($('uncertainBar')) {
    $('uncertainBar').style.width =
      `${consensus.uncertainty}%`;
  }

  if ($('engineBadge')) {
    if (
      consensus.activeModels === 1
    ) {
      $('engineBadge').textContent =
        'V6.1 • LIMITED EVIDENCE';

    } else if (
      consensus.verdict ===
      'CONFLICTING EVIDENCE'
    ) {
      $('engineBadge').textContent =
        'V6.1 • CONFLICT DEFENSE';

    } else if (
      consensus.calibrationReady
    ) {
      $('engineBadge').textContent =
        'V6.1 • BENCHMARK CALIBRATED';

    } else if (
      consensus.thirdUsed
    ) {
      $('engineBadge').textContent =
        'V6.1 • 3-MODEL CONSENSUS';

    } else if (
      isMobileDevice()
    ) {
      $('engineBadge').textContent =
        'V6.1 • MOBILE SAFE';

    } else {
      $('engineBadge').textContent =
        'V6.1 • SMART CONSENSUS';
    }
  }


  /* =======================================================
     EVIDENCE
  ======================================================= */

  const humanReasons =
    human.reasons.length
      ? human.reasons
          .slice(0, 5)
          .join(' • ')
      : 'No strong human-style counter-signals';

  const modernText =
    Number.isFinite(
      scores.modern
    )
      ? `${scores.modern}% AI signal`
      : isMobileDevice()
        ? 'Disabled on mobile for stability'
        : 'Not used / unavailable';

  const e5Text =
    Number.isFinite(
      scores.e5
    )
      ? `${scores.e5}% AI signal`
      : 'Unavailable';

  const tmrText =
    Number.isFinite(
      scores.tmr
    )
      ? `${scores.tmr}% AI signal`
      : 'Unavailable';

  const outlierText =
    consensus.outlier?.detected
      ? `${String(
          consensus.outlier.detector
        ).toUpperCase()} (${consensus.outlier.value}%) differs by ${Math.round(
          consensus.outlier.outlierDistance
        )} points from the agreeing pair.`
      : 'No clear detector outlier';

  const modelAgreementText =
    `${consensus.agreement.aiModels}/${consensus.agreement.active} models ≥70% AI · ` +
    `${consensus.agreement.humanModels}/${consensus.agreement.active} models ≤30% AI`;

  const evidence = [
    [
      'Final decision',
      consensus.verdict,
      'Outcome'
    ],

    [
      'Diagnostic AI signal',
      `${consensus.calibrated}%`,
      resolvedDecision
        ? 'Resolved'
        : 'Not a probability'
    ],

    [
      'Raw detector median',
      `${consensus.raw}%`,
      'Diagnostic'
    ],

    [
      'Reliability-weighted signal',
      `${consensus.weighted}%`,
      'Diagnostic'
    ],

    [
      'Evidence quality',
      `${consensus.quality}%`,
      consensus.quality >= 75
        ? 'Strong'
        : consensus.quality >= 50
          ? 'Medium'
          : 'Weak'
    ],

    [
      'Evidence sufficiency',
      `${consensus.activeModels}/3 independent models active`,
      consensus.activeModels >= 2
        ? 'Sufficient model count'
        : 'Insufficient'
    ],

    [
      'Model agreement',
      modelAgreementText,
      consensus.agreement.aiStrong ||
      consensus.agreement.humanStrong
        ? 'Strong'
        : 'Mixed'
    ],

    [
      'Outlier analysis',
      outlierText,
      consensus.outlier?.detected
        ? 'Possible outlier'
        : 'Clear'
    ],

    [
      'TMR detector',
      tmrText,
      'Model A'
    ],

    [
      'E5-small detector',
      e5Text,
      'Model B'
    ],

    [
      'ModernBERT judge',
      modernText,
      'Model C'
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
      'Raw model spread',
      `${consensus.rawSpread} points`,
      consensus.rawSpread >= 45
        ? 'Large'
        : 'Acceptable'
    ],

    [
      'Effective conflict',
      `${consensus.effectiveConflict} points`,
      consensus.effectiveConflict >= 38
        ? 'High'
        : 'Acceptable'
    ],

    [
      'Segment confirmation',
      `Mean ${consensus.segmentMean}% · range ${consensus.segmentRange} points`,
      consensus.strongSegmentAI
        ? 'AI-supporting'
        : consensus.strongSegmentHuman
          ? 'Human-supporting'
          : 'Mixed'
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      (
        domain === 'books' ||
        domain === 'poetry'
      )
        ? 'Protected domain'
        : 'Context'
    ],

    [
      'Calibration',
      consensus.calibrationReady
        ? `Active · AI ≥${consensus.aiThreshold}% · Human ≤${consensus.humanThreshold}% · Quality ≥${consensus.minQuality}%`
        : 'Not enough benchmark data yet',
      consensus.calibrationReady
        ? 'Active'
        : 'Pending'
    ],

    [
      'Language fit',
      language === 'English'
        ? 'English — strongest supported path'
        : 'Non-English — reduced reliability',
      'Context'
    ]
  ];

  if ($('evidence')) {
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


  /* =======================================================
     DOCUMENT METRICS
  ======================================================= */

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

    'Mobile safe mode':
      isMobileDevice()
        ? 'Yes'
        : 'No',

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

    'Raw detector median':
      `${consensus.raw}%`,

    'Weighted signal':
      `${consensus.weighted}%`,

    'Diagnostic calibrated signal':
      `${consensus.calibrated}%`,

    'Raw model spread':
      `${consensus.rawSpread} pts`,

    'Effective conflict':
      `${consensus.effectiveConflict} pts`,

    'Model deviation':
      consensus.modelSD,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Evidence quality':
      `${consensus.quality}%`,

    Outlier:
      consensus.outlier?.detected
        ? consensus.outlier.detector
        : 'None',

    'Strong AI agreement':
      consensus.agreement.aiStrong
        ? 'Yes'
        : 'No',

    'Strong human agreement':
      consensus.agreement.humanStrong
        ? 'Yes'
        : 'No',

    'Strong segment AI':
      consensus.strongSegmentAI
        ? 'Yes'
        : 'No',

    'Strong segment human':
      consensus.strongSegmentHuman
        ? 'Yes'
        : 'No',

    'Calibration ready':
      consensus.calibrationReady
        ? 'Yes'
        : 'No',

    Decision:
      consensus.verdict
  };

  if ($('metrics')) {
    $('metrics').innerHTML =
      Object.entries(metrics)
        .map(
          ([key, value]) => `
<div class="metric">
  <span>${escapeHTML(key)}</span>
  <b>${escapeHTML(String(value))}</b>
</div>
`
        )
        .join('');
  }


  /* =======================================================
     TRACE MAP
  ======================================================= */

  const chunks =
    chunkText(
      textEl?.value
        ?.trim() ||
      ''
    );

  if ($('segments')) {
    $('segments').innerHTML =
      chunks
        .map(
          (
            chunk,
            index
          ) => {
            const score =
              segmentScores[index] ??
              50;

            return `
<div class="segment">

  <div class="segmentHead">
    <b>Segment ${index + 1}</b>
    <span>${score}% TMR diagnostic</span>
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

  $('report')
    ?.scrollIntoView({
      behavior:
        'smooth',

      block:
        'start'
    });

  renderBenchmarkPanel();
}


/* =========================================================
   BENCHMARK PROMPT
========================================================= */

function benchmarkPrompt(scan) {
  const answer =
    prompt(
`AI TRACE V6.1 BENCHMARK

Only label samples whose TRUE origin you KNOW.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known human + AI mixture
UNKNOWN = origin not known

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
      'Benchmark skipped. Use AI, HUMAN, MIXED or UNKNOWN.'
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

  const prefixMap = {
    AI:
      'A',

    HUMAN:
      'H',

    MIXED:
      'M',

    UNKNOWN:
      'U'
  };

  const prefix =
    prefixMap[
      truth
    ];

  const sameTruthCount =
    records.filter(
      record =>
        record.truth ===
        truth
    ).length;

  const id =
    `${prefix}-${String(
      sameTruthCount + 1
    ).padStart(
      3,
      '0'
    )}`;

  /*
    IMPORTANT:
    Prediction is frozen BEFORE ground truth
    is allowed to affect later calibration.
  */

  const record = {
    id,

    truth,

    source,

    savedAt:
      new Date()
        .toISOString(),

    predictionFrozen:
      true,

    ...scan
  };

  records.push(
    record
  );

  saveBench(
    records
  );

  renderBenchmarkPanel();

  const binary =
    binaryRecords();

  const metrics =
    evaluatePredictions(
      binary,
      prediction
    );

  const readiness =
    calibrationReadiness();

  alert(
`Benchmark saved: ${id}

Known AI: ${metrics.totalAI}
Known HUMAN: ${metrics.totalHuman}

Coverage: ${metrics.coverage}%
Selective accuracy: ${metrics.selectiveAccuracy}%

False-positive rate: ${metrics.fpr}%
False-negative rate: ${metrics.fnr}%

AI abstention rate: ${metrics.aiAbstainRate}%
Human abstention rate: ${metrics.humanAbstainRate}%

Calibration: ${readiness.message}

Development measurements only.`
  );
}


/* =========================================================
   BENCHMARK INSPECTORS
========================================================= */

function falsePositives() {
  return binaryRecords()
    .filter(
      record =>
        record.truth ===
          'HUMAN' &&
        prediction(record) ===
          'AI'
    );
}

function falseNegatives() {
  return binaryRecords()
    .filter(
      record =>
        record.truth ===
          'AI' &&
        prediction(record) ===
          'HUMAN'
    );
}

function abstentions() {
  return binaryRecords()
    .filter(
      record =>
        prediction(record) ===
        'ABSTAIN'
    );
}

function mixedRecords() {
  return loadBench()
    .filter(
      record =>
        record.truth ===
        'MIXED'
    );
}


/* =========================================================
   BENCHMARK LAB
========================================================= */

function renderBenchmarkPanel() {
  const report =
    $('report');

  if (!report) {
    return;
  }

  let panel =
    $('benchmarkPanelV61');

  if (!panel) {
    panel =
      document.createElement(
        'section'
      );

    panel.id =
      'benchmarkPanelV61';

    panel.className =
      'panel devPanel';

    panel.style.marginTop =
      '18px';

    report.appendChild(
      panel
    );
  }

  const rows =
    binaryRecords();

  const ensemble =
    evaluatePredictions(
      rows,
      prediction
    );

  const reliability =
    detectorReliability();

  const domains =
    domainPerformance();

  const readiness =
    calibrationReadiness();

  const calibration =
    searchCalibration();

  const fp =
    falsePositives();

  const fn =
    falseNegatives();

  const abstain =
    abstentions();

  const mixed =
    mixedRecords();

  const detectorHTML =
    [
      'tmr',
      'e5',
      'modern'
    ]
      .map(
        name => {
          const info =
            reliability[
              name
            ];

          const m =
            info.metrics;

          return `
<div class="ev">

  <div class="evTop">
    <span>${escapeHTML(
      name.toUpperCase()
    )}</span>

    <span>
      ${info.samples} samples
    </span>
  </div>

  <small>
    Reliability ${info.weight}
    · Accuracy ${m.selectiveAccuracy}%
    · Coverage ${m.coverage}%
    · FPR ${m.fpr}%
    · FNR ${m.fnr}%
  </small>

</div>
`;
        }
      )
      .join('');

  const domainHTML =
    Object.entries(
      domains
    )
      .map(
        (
          [
            domain,
            metrics
          ]
        ) => `
<div class="metric">

  <span>
    ${escapeHTML(domain)}
  </span>

  <b>
    ${metrics.total} samples
    · Acc ${metrics.selectiveAccuracy}%
    · Coverage ${metrics.coverage}%
    · FPR ${metrics.fpr}%
    · FNR ${metrics.fnr}%
  </b>

</div>
`
      )
      .join('');

  function recordCard(
    record
  ) {
    return `
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
      prediction(record)
    )}
    · Signal ${escapeHTML(
      String(
        record.consensus
          ?.calibrated ??
        '?'
      )
    )}%
    · Quality ${escapeHTML(
      String(
        record.consensus
          ?.quality ??
        '?'
      )
    )}%
    · ${escapeHTML(
      record.consensus
        ?.verdict ||
      'legacy'
    )}
  </small>

</div>
`;
  }

  const fpHTML =
    fp
      .slice()
      .reverse()
      .slice(
        0,
        10
      )
      .map(
        recordCard
      )
      .join('');

  const fnHTML =
    fn
      .slice()
      .reverse()
      .slice(
        0,
        10
      )
      .map(
        recordCard
      )
      .join('');

  const abstainHTML =
    abstain
      .slice()
      .reverse()
      .slice(
        0,
        10
      )
      .map(
        recordCard
      )
      .join('');

  const recentHTML =
    loadBench()
      .slice()
      .reverse()
      .slice(
        0,
        12
      )
      .map(
        recordCard
      )
      .join('');

  let calibrationHTML = '';

  if (
    calibration.ready
  ) {
    calibrationHTML = `
<div class="ev">

  <div class="evTop">
    <span>Experimental calibration</span>
    <span>ACTIVE</span>
  </div>

  <small>
    AI threshold ${calibration.aiThreshold}%
    · Human threshold ${calibration.humanThreshold}%
    · Minimum quality ${calibration.minQuality}%
    · Coverage ${calibration.coverage}%
    · Selective accuracy ${calibration.selectiveAccuracy}%
  </small>

</div>
`;

  } else {
    calibrationHTML = `
<div class="ev">

  <div class="evTop">
    <span>Experimental calibration</span>
    <span>COLLECTING DATA</span>
  </div>

  <small>
    Need at least 20 binary samples,
    including at least 8 known AI,
    8 known HUMAN and 2 domains.
  </small>

</div>
`;
  }

  panel.innerHTML = `

<span class="over">
  V6.1 BENCHMARK LAB • DEVELOPMENT ONLY
</span>

<h2>
  Reliability & Evidence Lab
</h2>

<p class="sub">
  Known samples evaluate the detector. Abstentions remain separate from errors. Detector scores are diagnostic signals and are not treated as proof of authorship.
</p>


<h3>
  Calibration readiness
</h3>

<div class="evidence">

  <div class="ev">

    <div class="evTop">

      <span>
        ${readiness.message}
      </span>

      <span>
        ${
          readiness.ready
            ? 'READY'
            : 'COLLECTING'
        }
      </span>

    </div>

    <small>
      ${readiness.total} binary samples
      · ${readiness.ai} AI
      · ${readiness.human} HUMAN
      · ${readiness.domains} domains
      · ${mixed.length} MIXED
    </small>

  </div>

  ${calibrationHTML}

</div>


<h3>
  Confusion matrix
</h3>

<div class="metrics">

  <div class="metric">
    <span>True positive</span>
    <b>${ensemble.TP}</b>
  </div>

  <div class="metric">
    <span>True negative</span>
    <b>${ensemble.TN}</b>
  </div>

  <div class="metric">
    <span>False positive</span>
    <b>${ensemble.FP}</b>
  </div>

  <div class="metric">
    <span>False negative</span>
    <b>${ensemble.FN}</b>
  </div>

  <div class="metric">
    <span>AI abstentions</span>
    <b>${ensemble.aiAbstain}</b>
  </div>

  <div class="metric">
    <span>Human abstentions</span>
    <b>${ensemble.humanAbstain}</b>
  </div>

</div>


<h3>
  Ensemble performance
</h3>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${ensemble.total}</b>
  </div>

  <div class="metric">
    <span>Coverage</span>
    <b>${ensemble.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${ensemble.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${ensemble.precision}%</b>
  </div>

  <div class="metric">
    <span>AI recall</span>
    <b>${ensemble.recall}%</b>
  </div>

  <div class="metric">
    <span>Human specificity</span>
    <b>${ensemble.specificity}%</b>
  </div>

  <div class="metric">
    <span>False-positive rate</span>
    <b>${ensemble.fpr}%</b>
  </div>

  <div class="metric">
    <span>False-negative rate</span>
    <b>${ensemble.fnr}%</b>
  </div>

  <div class="metric">
    <span>AI abstention rate</span>
    <b>${ensemble.aiAbstainRate}%</b>
  </div>

  <div class="metric">
    <span>Human abstention rate</span>
    <b>${ensemble.humanAbstainRate}%</b>
  </div>

</div>


<h3>
  Detector reliability
</h3>

<div class="evidence">
  ${detectorHTML}
</div>


<h3>
  Domain performance
</h3>

<div class="metrics">

  ${
    domainHTML ||
    `
    <div class="metric">
      <span>No domain benchmark data</span>
      <b>—</b>
    </div>
    `
  }

</div>


<h3>
  False-positive inspector
</h3>

<div class="evidence">

  ${
    fpHTML ||
    `
    <div class="ev">
      <small>
        No known HUMAN sample has been classified as AI.
      </small>
    </div>
    `
  }

</div>


<h3>
  False-negative inspector
</h3>

<div class="evidence">

  ${
    fnHTML ||
    `
    <div class="ev">
      <small>
        No known AI sample has been classified as HUMAN.
      </small>
    </div>
    `
  }

</div>


<h3>
  Abstention inspector
</h3>

<div class="evidence">

  ${
    abstainHTML ||
    `
    <div class="ev">
      <small>
        No binary abstentions recorded.
      </small>
    </div>
    `
  }

</div>


<h3>
  Recent benchmark records
</h3>

<div class="evidence">

  ${
    recentHTML ||
    `
    <div class="ev">
      <small>
        No benchmark records yet.
      </small>
    </div>
    `
  }

</div>
`;
}


/* =========================================================
   HISTORY
========================================================= */

function saveHistory(
  scan
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

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(
        history.slice(
          -100
        )
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


/* =========================================================
   EXPORT JSON
========================================================= */

function exportBenchmarkJSON() {
  const data =
    window.AITraceV61
      .benchmark();

  const blob =
    new Blob(
      [
        JSON.stringify(
          data,
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
    `AI-Trace-V61-Benchmark-${Date.now()}.json`;

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


/* =========================================================
   EXPORT CSV
========================================================= */

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
    'raw',
    'weighted',
    'calibrated',
    'quality',
    'confidence',
    'uncertainty',
    'verdict',
    'rawSpread',
    'effectiveConflict',
    'segmentMean',
    'segmentRange'
  ];

  const csvEscape =
    value => {
      const stringValue =
        String(
          value ??
          ''
        );

      if (
        /[",\n]/.test(
          stringValue
        )
      ) {
        return `"${stringValue.replace(
          /"/g,
          '""'
        )}"`;
      }

      return stringValue;
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

        record.consensus?.weighted,

        record.consensus?.calibrated,

        record.consensus?.quality,

        record.consensus?.confidence,

        record.consensus?.uncertainty,

        record.consensus?.verdict,

        record.consensus?.rawSpread,

        record.consensus?.effectiveConflict,

        record.consensus?.segmentMean,

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
    `AI-Trace-V61-Benchmark-${Date.now()}.csv`;

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


/* =========================================================
   DEVELOPER API
========================================================= */

window.AITraceV61 = {
  benchmark() {
    const rows =
      binaryRecords();

    return {
      version:
        VERSION,

      readiness:
        calibrationReadiness(),

      calibration:
        searchCalibration(),

      ensemble:
        evaluatePredictions(
          rows,
          prediction
        ),

      detectors:
        detectorReliability(),

      domains:
        domainPerformance(),

      inspectors: {
        falsePositives:
          falsePositives(),

        falseNegatives:
          falseNegatives(),

        abstentions:
          abstentions(),

        mixed:
          mixedRecords()
      },

      records:
        loadBench()
    };
  },

  history() {
    try {
      return JSON.parse(
        localStorage.getItem(
          HISTORY_KEY
        ) || '[]'
      );

    } catch {
      return [];
    }
  },

  exportJSON() {
    exportBenchmarkJSON();
  },

  exportCSV() {
    exportBenchmarkCSV();
  },

  clearBenchmark() {
    const confirmation =
      confirm(
        'Delete all V6.1 benchmark records from this device?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      BENCH_KEY
    );

    renderBenchmarkPanel();

    alert(
      'V6.1 benchmark data deleted.'
    );
  },

  clearHistory() {
    const confirmation =
      confirm(
        'Delete local V6.1 scan history?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      HISTORY_KEY
    );

    alert(
      'V6.1 history deleted.'
    );
  }
};


/* =========================================================
   INITIALIZE
========================================================= */

updateCount();

setTimeout(
  renderBenchmarkPanel,
  350
);

console.info(
  `AI TRACE V${VERSION} loaded`
);
