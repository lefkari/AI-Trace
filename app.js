/*
  AI TRACE V4.1
  Detection + Consensus + Benchmark Engine

  One-file build.
  No benchmark.js dependency required.
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

const APP_VERSION = '4.1';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCHMARK_STORAGE =
  'aiTraceBenchmarkV41';

const SCAN_HISTORY_STORAGE =
  'aiTraceScanHistoryV41';

let tmrClassifier = null;
let modernClassifier = null;

/* =========================================================
   DOM
========================================================= */

const $ = id =>
  document.getElementById(id);

const text = $('text');

/* =========================================================
   BASIC UI
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

/* =========================================================
   PROGRESS
========================================================= */

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
      percent + '%';

  $('progressText')
    .textContent =
      label;
}

/* =========================================================
   LANGUAGE DETECTION
========================================================= */

function detectLanguage(value) {

  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) || []
    ).length;

  const totalLetters =
    (
      value.match(
        /\p{L}/gu
      ) || []
    ).length;

  if (!totalLetters) {
    return 'Unknown';
  }

  return (
    latin /
    totalLetters
  ) > 0.80
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

  const averageSentenceLength =
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

  const sentenceLengths =
    sentences.map(
      sentence =>
        sentence
          .split(/\s+/)
          .filter(Boolean)
          .length
    );

  const meanSentenceLength =
    sentenceLengths
      .reduce(
        (a, b) =>
          a + b,
        0
      ) /
    Math.max(
      1,
      sentenceLengths.length
    );

  const variance =
    sentenceLengths
      .reduce(
        (
          sum,
          length
        ) =>
          sum +
          (
            length -
            meanSentenceLength
          ) ** 2,
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

    words:
      words.length,

    sentences:
      sentences.length,

    averageSentenceLength,

    lexicalDiversity,

    variance,

    transitions
  };
}

/* =========================================================
   CHUNKING
========================================================= */

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

/* =========================================================
   MODEL LOADERS
========================================================= */

async function loadTMR() {

  if (
    tmrClassifier
  ) {

    return tmrClassifier;
  }

  $('modelState')
    .textContent =
      'Loading TMR engine…';

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

async function loadModernBERT() {

  if (
    modernClassifier
  ) {

    return modernClassifier;
  }

  $('modelState')
    .textContent =
      'Loading Deep Scan engine…';

  progress(
    55,
    'Loading second detector…'
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

/* =========================================================
   MODEL OUTPUT
========================================================= */

function extractAIProbability(
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
    const result
    of results
  ) {

    const label =
      String(
        result.label ||
        ''
      ).toLowerCase();

    const score =
      Number(
        result.score
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

/* =========================================================
   STATISTICS
========================================================= */

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

function standardDeviation(
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

/* =========================================================
   SUPPORTING HEURISTIC
========================================================= */

function heuristicScore(
  profile
) {

  let score =
    42;

  if (
    profile.variance <
      35
  ) {

    score +=
      10;
  }

  if (
    profile.transitions >
      2
  ) {

    score +=
      8;
  }

  if (
    profile
      .averageSentenceLength >
      18 &&
    profile
      .averageSentenceLength <
      32
  ) {

    score +=
      8;
  }

  return Math.min(
    78,
    score
  );
}

/* =========================================================
   CONSENSUS ENGINE
========================================================= */

function buildConsensus({
  tmr,
  modern,
  segmentScores,
  profile,
  language,
  tmrWorked,
  modernWorked
}) {

  const activeModels =
    Number(
      tmrWorked
    ) +
    Number(
      modernWorked
    );

  const modelGap =
    (
      tmrWorked &&
      modernWorked
    )
      ? Math.abs(
          tmr -
          modern
        )
      : 0;

  const segmentDeviation =
    standardDeviation(
      segmentScores
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

  /*
    Raw detector signal.

    IMPORTANT:
    This is NOT interpreted as
    "probability that X% was AI".
  */

  let rawSignal;

  if (
    tmrWorked &&
    modernWorked
  ) {

    rawSignal =
      Math.round(
        tmr * 0.48 +
        modern * 0.48 +
        heuristicScore(
          profile
        ) * 0.04
      );

  } else if (
    tmrWorked
  ) {

    rawSignal =
      tmr;

  } else if (
    modernWorked
  ) {

    rawSignal =
      modern;

  } else {

    rawSignal =
      heuristicScore(
        profile
      );
  }

  /*
    Evidence uncertainty.
  */

  let uncertainty =
    10;

  if (
    activeModels <
      2
  ) {

    uncertainty +=
      35;
  }

  uncertainty +=
    Math.min(
      30,
      modelGap *
        0.60
    );

  uncertainty +=
    Math.min(
      30,
      segmentDeviation *
        0.65
    );

  if (
    profile.words <
      150
  ) {

    uncertainty +=
      12;
  }

  if (
    language !==
      'English'
  ) {

    uncertainty +=
      25;
  }

  uncertainty =
    Math.min(
      95,
      Math.round(
        uncertainty
      )
    );

  const confidence =
    Math.max(
      5,
      100 -
      uncertainty
    );

  /*
    Verdict safeguards.
  */

  const modelsConflict =
    (
      tmrWorked &&
      modernWorked &&
      modelGap >= 35
    );

  const segmentsHighlyUnstable =
    (
      segmentDeviation >=
        28 ||
      segmentRange >=
        70
    );

  const limitedEvidence =
    activeModels <
      2;

  let verdict =
    'INCONCLUSIVE';

  if (
    modelsConflict ||
    segmentsHighlyUnstable ||
    limitedEvidence
  ) {

    verdict =
      'INCONCLUSIVE';

  } else if (
    rawSignal >=
      85
  ) {

    verdict =
      'Strong AI evidence';

  } else if (
    rawSignal >=
      65
  ) {

    verdict =
      'Likely AI';

  } else if (
    rawSignal <=
      15
  ) {

    verdict =
      'Strong human evidence';

  } else if (
    rawSignal <=
      35
  ) {

    verdict =
      'Likely human';

  } else {

    verdict =
      'INCONCLUSIVE';
  }

  return {

    rawSignal,

    uncertainty,

    confidence,

    verdict,

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

    activeModels,

    modelsConflict,

    segmentsHighlyUnstable
  };
}

/* =========================================================
   BENCHMARK STORAGE
========================================================= */

function loadBenchmark() {

  try {

    const raw =
      localStorage.getItem(
        BENCHMARK_STORAGE
      );

    return raw
      ? JSON.parse(
          raw
        )
      : [];

  } catch {

    return [];
  }
}

function saveBenchmark(
  records
) {

  try {

    localStorage.setItem(
      BENCHMARK_STORAGE,
      JSON.stringify(
        records
      )
    );

  } catch (
    error
  ) {

    console.warn(
      'Benchmark save failed',
      error
    );
  }
}

function nextBenchmarkID(
  truth,
  records
) {

  const prefix =
    truth === 'AI'
      ? 'A'
      : 'H';

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

function addBenchmark({
  groundTruth,
  source,
  profile,
  language,
  tmr,
  modern,
  tmrWorked,
  modernWorked,
  segmentScores,
  consensus
}) {

  const records =
    loadBenchmark();

  const id =
    nextBenchmarkID(
      groundTruth,
      records
    );

  records.push({

    id,

    appVersion:
      APP_VERSION,

    timestamp:
      new Date()
        .toISOString(),

    groundTruth,

    source,

    words:
      profile.words,

    language,

    models: {

      tmr,

      modern,

      tmrWorked,

      modernWorked
    },

    segments:
      segmentScores,

    rawSignal:
      consensus.rawSignal,

    verdict:
      consensus.verdict,

    confidence:
      consensus.confidence,

    uncertainty:
      consensus.uncertainty,

    modelGap:
      consensus.modelGap,

    segmentDeviation:
      consensus.segmentDeviation,

    segmentRange:
      consensus.segmentRange
  });

  saveBenchmark(
    records
  );

  return id;
}

/* =========================================================
   BASELINE BENCHMARK PREDICTION

   These thresholds are evaluation
   baselines only — not final calibration.
========================================================= */

function benchmarkPrediction(
  record
) {

  const signal =
    record.rawSignal;

  if (
    signal >= 70
  ) {

    return 'AI';
  }

  if (
    signal <= 30
  ) {

    return 'HUMAN';
  }

  return 'UNCERTAIN';
}

/* =========================================================
   BENCHMARK METRICS
========================================================= */

function calculateBenchmarkMetrics() {

  const records =
    loadBenchmark()
      .filter(
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
    of records
  ) {

    const prediction =
      benchmarkPrediction(
        record
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

  const falsePositiveRate =
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

  const falseNegativeRate =
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

    total:
      records.length,

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
        falsePositiveRate *
        100
      ),

    falseNegativeRate:
      Math.round(
        falseNegativeRate *
        100
      )
  };
}

/* =========================================================
   CALIBRATION TABLE
========================================================= */

function buildCalibrationTable() {

  const records =
    loadBenchmark()
      .filter(
        record =>
          record.groundTruth ===
            'AI' ||
          record.groundTruth ===
            'HUMAN'
      );

  const bins = [];

  for (
    let start = 0;
    start < 100;
    start += 10
  ) {

    const end =
      start + 10;

    const samples =
      records.filter(
        record =>
          record.rawSignal >=
            start &&
          (
            start === 90
              ? record.rawSignal <=
                  100
              : record.rawSignal <
                  end
          )
      );

    if (
      !samples.length
    ) {

      continue;
    }

    const actualAI =
      samples.filter(
        sample =>
          sample.groundTruth ===
            'AI'
      ).length;

    bins.push({

      range:
        `${start}-${end}%`,

      samples:
        samples.length,

      actualAIRate:
        Math.round(
          (
            actualAI /
            samples.length
          ) *
          100
        )
    });
  }

  return bins;
}

/* =========================================================
   SCAN HISTORY
========================================================= */

function saveScanHistory(
  result
) {

  try {

    const history =
      JSON.parse(
        localStorage.getItem(
          SCAN_HISTORY_STORAGE
        ) || '[]'
      );

    history.push({

      timestamp:
        new Date()
          .toISOString(),

      ...result
    });

    localStorage.setItem(
      SCAN_HISTORY_STORAGE,
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
      'Scan history save failed',
      error
    );
  }
}

/* =========================================================
   BENCHMARK PROMPT
========================================================= */

function askBenchmarkLabel(
  scanData
) {

  /*
    Temporary developer benchmark workflow.

    Cancel or empty = skip.
  */

  const answer =
    prompt(
`AI TRACE BENCHMARK

Do you KNOW the true origin of this text?

Type:
AI      = definitely AI-generated
HUMAN   = definitely human-written

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
      'Benchmark not saved. Use only AI or HUMAN.'
    );

    return;
  }

  const source =
    prompt(
      'Source / note for this benchmark sample:',
      truth === 'AI'
        ? 'Known AI-generated sample'
        : 'Known human-written sample'
    ) || '';

  const id =
    addBenchmark({

      groundTruth:
        truth,

      source,

      ...scanData
    });

  const metrics =
    calculateBenchmarkMetrics();

  alert(
`Benchmark saved: ${id}

Total known samples: ${metrics.total}

TP: ${metrics.TP}
TN: ${metrics.TN}
FP: ${metrics.FP}
FN: ${metrics.FN}
Uncertain: ${metrics.uncertain}

Accuracy*: ${metrics.accuracy}%
Precision*: ${metrics.precision}%
Recall*: ${metrics.recall}%
Specificity*: ${metrics.specificity}%

False Positive Rate*: ${metrics.falsePositiveRate}%
False Negative Rate*: ${metrics.falseNegativeRate}%

*These numbers are experimental until the benchmark dataset becomes much larger.`
  );
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

  /* ==============================
     TMR
  ============================== */

  try {

    const tmr =
      await loadTMR();

    progress(
      20,
      'TMR Quick Scan…'
    );

    tmrDocument =
      await classify(
        tmr,
        value
      );

    for (
      let index = 0;
      index < chunks.length;
      index++
    ) {

      progress(
        25 +
        Math.round(
          (
            index /
            Math.max(
              1,
              chunks.length
            )
          ) *
          25
        ),
        `TMR segment ${index + 1}/${chunks.length}`
      );

      const score =
        await classify(
          tmr,
          chunks[index]
        );

      tmrSegments.push(
        score
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

    tmrDocument =
      heuristicScore(
        profile
      );

    for (
      let index = 0;
      index < chunks.length;
      index++
    ) {

      tmrSegments.push(
        tmrDocument
      );
    }
  }

  /* ==============================
     MODERNBERT
  ============================== */

  try {

    const modern =
      await loadModernBERT();

    progress(
      70,
      'ModernBERT Deep Scan…'
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

    modernDocument =
      50;
  }

  /* ==============================
     CONSENSUS
  ============================== */

  progress(
    88,
    'Building evidence consensus…'
  );

  const consensus =
    buildConsensus({

      tmr:
        tmrDocument,

      modern:
        modernDocument,

      segmentScores:
        tmrSegments,

      profile,

      language,

      tmrWorked,

      modernWorked
    });

  /* ==============================
     RENDER
  ============================== */

  renderV41({

    consensus,

    profile,

    chunks,

    segmentScores:
      tmrSegments,

    language,

    tmrDocument,

    modernDocument,

    tmrWorked,

    modernWorked
  });

  /* ==============================
     LOCAL SCAN HISTORY
  ============================== */

  saveScanHistory({

    version:
      APP_VERSION,

    words:
      profile.words,

    language,

    tmr:
      tmrDocument,

    modern:
      modernDocument,

    rawSignal:
      consensus.rawSignal,

    verdict:
      consensus.verdict,

    confidence:
      consensus.confidence,

    uncertainty:
      consensus.uncertainty,

    modelGap:
      consensus.modelGap,

    segmentDeviation:
      consensus.segmentDeviation,

    segmentRange:
      consensus.segmentRange
  });

  progress(
    100,
    'Trace complete'
  );

  $('modelState')
    .textContent =
      (
        tmrWorked &&
        modernWorked
      )
        ? 'V4.1 consensus engine ready ✓'
        : 'Limited evidence mode';

  setTimeout(
    () => {

      $('progress')
        .classList
        .add('hidden');

    },
    500
  );

  $('scan').disabled =
    false;

  /*
    Temporary ground-truth collection.
  */

  setTimeout(
    () => {

      askBenchmarkLabel({

        profile,

        language,

        tmr:
          tmrDocument,

        modern:
          modernDocument,

        tmrWorked,

        modernWorked,

        segmentScores:
          tmrSegments,

        consensus
      });

    },
    650
  );
}

/* =========================================================
   HTML ESCAPE
========================================================= */

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

/* =========================================================
   RENDER V4.1
========================================================= */

function renderV41({
  consensus,
  profile,
  chunks,
  segmentScores,
  language,
  tmrDocument,
  modernDocument,
  tmrWorked,
  modernWorked
}) {

  $('report')
    .classList
    .remove('hidden');

  /*
    IMPORTANT:
    The UI still contains the old
    "AI INVOLVEMENT" label from V3.

    During final design we will rename
    this to "AI DETECTION SIGNAL".
  */

  $('score')
    .textContent =
      consensus.rawSignal +
      '%';

  $('scaleFill')
    .style
    .width =
      consensus.rawSignal +
      '%';

  $('verdict')
    .textContent =
      consensus.verdict;

  const confidenceLabel =
    consensus.confidence >=
      70
      ? 'High'
      : consensus.confidence >=
          45
        ? 'Medium'
        : 'Low';

  $('confidence')
    .textContent =
      `Confidence: ${confidenceLabel} (${consensus.confidence}%)`;

  if (
    consensus.verdict ===
      'INCONCLUSIVE'
  ) {

    $('explain')
      .textContent =
`AI Trace detected conflicting or unstable evidence and will not force an AI/Human classification.

Model disagreement: ${consensus.modelGap} points.
Segment deviation: ${consensus.segmentDeviation}.
Segment range: ${consensus.segmentRange} points.`;

  } else {

    $('explain')
      .textContent =
`AI Trace combined two independent detection engines.

TMR signal: ${tmrDocument}%.
ModernBERT signal: ${modernDocument}%.
Model disagreement: ${consensus.modelGap} points.`;
  }

  const humanDisplay =
    100 -
    consensus.rawSignal;

  $('humanVal')
    .textContent =
      humanDisplay +
      '%';

  $('aiVal')
    .textContent =
      consensus.rawSignal +
      '%';

  $('uncertainVal')
    .textContent =
      consensus.uncertainty +
      '%';

  $('humanBar')
    .style
    .width =
      humanDisplay +
      '%';

  $('aiBar')
    .style
    .width =
      consensus.rawSignal +
      '%';

  $('uncertainBar')
    .style
    .width =
      consensus.uncertainty +
      '%';

  $('engineBadge')
    .textContent =
      (
        tmrWorked &&
        modernWorked
      )
        ? 'V4.1 • 2-MODEL CONSENSUS'
        : 'LIMITED EVIDENCE';

  /* ==============================
     EVIDENCE
  ============================== */

  const evidence = [

    [
      'TMR detector',

      tmrWorked
        ? `${tmrDocument}% AI detection signal`
        : 'Detector unavailable',

      tmrWorked
        ? 'Model A'
        : 'Unavailable'
    ],

    [
      'ModernBERT detector',

      modernWorked
        ? `${modernDocument}% AI detection signal`
        : 'Detector unavailable',

      modernWorked
        ? 'Model B'
        : 'Unavailable'
    ],

    [
      'Model disagreement',

      `${consensus.modelGap} percentage points`,

      consensus.modelGap >=
        35
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment deviation',

      `${consensus.segmentDeviation}`,

      consensus.segmentDeviation >=
        28
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Segment range',

      `${consensus.segmentRange} percentage points`,

      consensus.segmentRange >=
        70
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Language fit',

      language ===
        'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English detected — reliability is reduced.',

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

  /* ==============================
     METRICS
  ============================== */

  const metrics = {

    Words:
      profile.words,

    Sentences:
      profile.sentences,

    'Avg. words / sentence':
      profile
        .averageSentenceLength
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
      `${consensus.activeModels}/2`,

    'Model disagreement':
      `${consensus.modelGap} pts`,

    'Segment deviation':
      consensus.segmentDeviation,

    'Segment range':
      `${consensus.segmentRange} pts`
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
              key,
              value
            ]
          ) => `
<div class="metric">
  <span>${key}</span>
  <b>${value}</b>
</div>`
        )
        .join('');

  /* ==============================
     TRACE MAP
  ============================== */

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

/* =========================================================
   DEVELOPER / BENCHMARK UTILITIES

   Available from browser console later if needed:
   AITraceBenchmark.report()
   AITraceBenchmark.exportJSON()
   AITraceBenchmark.clear()
========================================================= */

window.AITraceBenchmark = {

  report() {

    return {

      version:
        APP_VERSION,

      metrics:
        calculateBenchmarkMetrics(),

      calibration:
        buildCalibrationTable(),

      samples:
        loadBenchmark()
    };
  },

  exportJSON() {

    const report =
      this.report();

    const json =
      JSON.stringify(
        report,
        null,
        2
      );

    const blob =
      new Blob(
        [json],
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
  },

  clear() {

    const confirmation =
      confirm(
        'Delete all AI Trace benchmark results stored on this device?'
      );

    if (
      !confirmation
    ) {

      return;
    }

    localStorage.removeItem(
      BENCHMARK_STORAGE
    );

    alert(
      'Benchmark data deleted.'
    );
  },

  history() {

    try {

      return JSON.parse(
        localStorage.getItem(
          SCAN_HISTORY_STORAGE
        ) || '[]'
      );

    } catch {

      return [];
    }
  }
};
