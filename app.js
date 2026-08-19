/*
  AI TRACE V4.2
  Robust Consensus + Calibration + Benchmark Engine
  Replace the entire app.js with this file.
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

const APP_VERSION = '4.2';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCHMARK_STORAGE =
  'aiTraceBenchmarkV42';

const LEGACY_BENCHMARK_STORAGE =
  'aiTraceBenchmarkV41';

const SCAN_HISTORY_STORAGE =
  'aiTraceScanHistoryV42';

const MIN_WORDS = 80;

const MIN_CALIBRATION_SAMPLES = 12;

const MIN_BIN_SAMPLES = 4;

let tmrClassifier = null;
let modernClassifier = null;


/* =========================================================
   DOM
========================================================= */

const $ = id =>
  document.getElementById(id);

const text = $('text');


/* =========================================================
   UPDATE OLD LABEL
========================================================= */

const scoreLabel =
  document.querySelector(
    '.scoreCard .over'
  );

if (scoreLabel) {

  scoreLabel.textContent =
    'AI DETECTION SIGNAL';
}


/* =========================================================
   BASIC UI
========================================================= */

text.oninput = () => {

  const words =
    text.value.trim()
      ? text.value
          .trim()
          .split(/\s+/)
          .length
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
      `${percent}%`;

  $('progressText')
    .textContent =
      label;
}


/* =========================================================
   LANGUAGE
========================================================= */

function detectLanguage(
  value
) {

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
        dtype: 'q4f16'
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


function median(
  values
) {

  if (
    !values.length
  ) {

    return 0;
  }

  const sorted =
    [...values]
      .sort(
        (a, b) =>
          a - b
      );

  const middle =
    Math.floor(
      sorted.length /
      2
    );

  if (
    sorted.length % 2
  ) {

    return sorted[
      middle
    ];
  }

  return (
    sorted[
      middle - 1
    ] +
    sorted[
      middle
    ]
  ) / 2;
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
    mean(
      values
    );

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
   SAFE JSON
========================================================= */

function safeParse(
  value,
  fallback = []
) {

  try {

    return JSON.parse(
      value
    );

  } catch {

    return fallback;
  }
}


/* =========================================================
   MIGRATE V4.1 BENCHMARK
========================================================= */

function migrateLegacyBenchmark() {

  const current =
    localStorage.getItem(
      BENCHMARK_STORAGE
    );

  if (
    current
  ) {

    return;
  }

  const legacy =
    localStorage.getItem(
      LEGACY_BENCHMARK_STORAGE
    );

  if (
    !legacy
  ) {

    return;
  }

  const records =
    safeParse(
      legacy,
      []
    );

  localStorage.setItem(
    BENCHMARK_STORAGE,
    JSON.stringify(
      records.map(
        record => ({
          ...record,
          migratedFrom:
            '4.1'
        })
      )
    )
  );
}


migrateLegacyBenchmark();


/* =========================================================
   BENCHMARK STORAGE
========================================================= */

function loadBenchmark() {

  return safeParse(
    localStorage.getItem(
      BENCHMARK_STORAGE
    ) || '[]',
    []
  );
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


/* =========================================================
   CALIBRATION DATA
========================================================= */

function calibrationRecords() {

  return loadBenchmark()
    .filter(
      record =>
        (
          record.groundTruth ===
            'AI' ||
          record.groundTruth ===
            'HUMAN'
        ) &&
        Number.isFinite(
          record.rawSignal
        )
    );
}


/* =========================================================
   LOCAL CALIBRATION
========================================================= */

function localCalibration(
  rawSignal
) {

  const records =
    calibrationRecords();

  if (
    records.length <
      MIN_CALIBRATION_SAMPLES
  ) {

    return {

      active:
        false,

      samples:
        records.length,

      calibrated:
        rawSignal,

      localAIRate:
        null
    };
  }

  const nearby =
    records.filter(
      record =>
        Math.abs(
          record.rawSignal -
          rawSignal
        ) <= 10
    );

  if (
    nearby.length <
      MIN_BIN_SAMPLES
  ) {

    return {

      active:
        false,

      samples:
        records.length,

      calibrated:
        rawSignal,

      localAIRate:
        null
    };
  }

  const aiCount =
    nearby.filter(
      record =>
        record.groundTruth ===
          'AI'
    ).length;

  const localAIRate =
    (
      aiCount /
      nearby.length
    ) *
    100;

  /*
    Do not allow a tiny benchmark
    to completely override the models.
  */

  const weight =
    Math.min(
      0.55,
      nearby.length /
      30
    );

  const calibrated =
    rawSignal *
      (
        1 -
        weight
      ) +
    localAIRate *
      weight;

  return {

    active:
      true,

    samples:
      records.length,

    nearby:
      nearby.length,

    calibrated:
      Math.round(
        calibrated
      ),

    localAIRate:
      Math.round(
        localAIRate
      )
  };
}


/* =========================================================
   ROBUST CONSENSUS ENGINE
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

  const segmentMedian =
    Math.round(
      median(
        segmentScores
      )
    );

  const segmentMean =
    Math.round(
      mean(
        segmentScores
      )
    );


  /* =====================================================
     RAW DETECTOR SIGNAL
  ===================================================== */

  let rawSignal;

  if (
    tmrWorked &&
    modernWorked
  ) {

    rawSignal =
      Math.round(

        tmr *
          0.43 +

        modern *
          0.43 +

        segmentMedian *
          0.10 +

        heuristicScore(
          profile
        ) *
          0.04
      );

  } else if (
    tmrWorked
  ) {

    rawSignal =
      Math.round(

        tmr *
          0.82 +

        segmentMedian *
          0.14 +

        heuristicScore(
          profile
        ) *
          0.04
      );

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

  rawSignal =
    clamp(
      rawSignal
    );


  /* =====================================================
     MODEL AGREEMENT
  ===================================================== */

  const modelAgreement =
    100 -
    clamp(
      modelGap *
        1.8
    );


  /* =====================================================
     SEGMENT STABILITY
  ===================================================== */

  const segmentStability =
    100 -
    clamp(

      segmentDeviation *
        1.55 +

      Math.max(
        0,
        segmentRange -
          45
      ) *
        0.45
    );


  /* =====================================================
     EVIDENCE QUALITY
  ===================================================== */

  let evidenceQuality =

    modelAgreement *
      0.45 +

    segmentStability *
      0.40 +

    (
      profile.words >=
        150
        ? 100
        : 65
    ) *
      0.10 +

    (
      language ===
        'English'
        ? 100
        : 55
    ) *
      0.05;


  if (
    activeModels <
      2
  ) {

    evidenceQuality *=
      0.60;
  }


  evidenceQuality =
    clamp(
      Math.round(
        evidenceQuality
      )
    );


  /* =====================================================
     BENCHMARK CALIBRATION
  ===================================================== */

  const calibration =
    localCalibration(
      rawSignal
    );


  const calibratedSignal =
    calibration.active
      ? calibration.calibrated
      : rawSignal;


  /* =====================================================
     RELIABILITY-ADJUSTED EVIDENCE INDEX

     Important:
     This is NOT "% of text written by AI".
  ===================================================== */

  const reliabilityWeight =
    evidenceQuality /
    100;


  const evidenceIndex =
    Math.round(

      50 +

      (
        calibratedSignal -
        50
      ) *

      reliabilityWeight
    );


  const uncertainty =
    100 -
    evidenceQuality;


  const confidence =
    evidenceQuality;


  /* =====================================================
     SAFETY CONDITIONS
  ===================================================== */

  const modelsConflict =
    modelGap >=
      35;


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


  /* =====================================================
     VERDICT
  ===================================================== */

  let verdict =
    'INCONCLUSIVE';


  if (
    limitedEvidence ||
    modelsConflict ||
    segmentsHighlyUnstable ||
    confidence <
      55
  ) {

    verdict =
      'INCONCLUSIVE';

  } else if (
    calibratedSignal >=
      85 &&
    confidence >=
      75
  ) {

    verdict =
      'Strong AI evidence';

  } else if (
    calibratedSignal >=
      67 &&
    confidence >=
      65
  ) {

    verdict =
      'Likely AI';

  } else if (
    calibratedSignal <=
      15 &&
    confidence >=
      75
  ) {

    verdict =
      'Strong human evidence';

  } else if (
    calibratedSignal <=
      33 &&
    confidence >=
      65
  ) {

    verdict =
      'Likely human';
  }


  return {

    rawSignal,

    calibratedSignal,

    evidenceIndex,

    uncertainty,

    confidence,

    verdict,

    modelGap:
      Math.round(
        modelGap
      ),

    modelAgreement:
      Math.round(
        modelAgreement
      ),

    segmentDeviation:
      Math.round(
        segmentDeviation
      ),

    segmentRange:
      Math.round(
        segmentRange
      ),

    segmentMedian,

    segmentMean,

    segmentStability:
      Math.round(
        segmentStability
      ),

    evidenceQuality,

    activeModels,

    modelsConflict,

    segmentsHighlyUnstable,

    calibration
  };
}


/* =========================================================
   ADD BENCHMARK
========================================================= */

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

    calibratedSignal:
      consensus.calibratedSignal,

    evidenceIndex:
      consensus.evidenceIndex,

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
      consensus.segmentRange,

    segmentMedian:
      consensus.segmentMedian,

    segmentMean:
      consensus.segmentMean
  });


  saveBenchmark(
    records
  );


  return id;
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


  const coverage =
    records.length
      ? decided /
        records.length
      : 0;


  return {

    total:
      records.length,

    decided,

    uncertain,

    coverage:
      Math.round(
        coverage *
        100
      ),

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
    calibrationRecords();

  const bins = [];


  for (
    let start = 0;
    start < 100;
    start += 10
  ) {

    const end =
      start +
      10;


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
      safeParse(
        localStorage.getItem(
          SCAN_HISTORY_STORAGE
        ) || '[]',
        []
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

  const answer =
    prompt(
`AI TRACE BENCHMARK

Do you KNOW the true origin of this text?

Type:

AI = definitely AI-generated

HUMAN = definitely human-written

Leave empty or Cancel to skip.`
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

Known samples: ${metrics.total}

Decided: ${metrics.decided}

Uncertain: ${metrics.uncertain}

Coverage: ${metrics.coverage}%

TP: ${metrics.TP}

TN: ${metrics.TN}

FP: ${metrics.FP}

FN: ${metrics.FN}

Accuracy*: ${metrics.accuracy}%

Precision*: ${metrics.precision}%

Recall*: ${metrics.recall}%

Specificity*: ${metrics.specificity}%

False Positive Rate*: ${metrics.falsePositiveRate}%

False Negative Rate*: ${metrics.falseNegativeRate}%

*Experimental until the benchmark dataset becomes much larger.`
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
    wordCount <
      MIN_WORDS
  ) {

    alert(
      `Paste at least ${MIN_WORDS} words for a meaningful analysis.`
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


  /* =====================================================
     TMR
  ===================================================== */

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
      index <
        chunks.length;
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
          chunks[
            index
          ]
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
      index <
        chunks.length;
      index++
    ) {

      tmrSegments.push(
        tmrDocument
      );
    }
  }


  /* =====================================================
     MODERN BERT
  ===================================================== */

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


  /* =====================================================
     CONSENSUS + CALIBRATION
  ===================================================== */

  progress(
    88,
    'Calibrating evidence…'
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


  /* =====================================================
     RENDER
  ===================================================== */

  renderV42({

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


  /* =====================================================
     HISTORY
  ===================================================== */

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

    calibratedSignal:
      consensus.calibratedSignal,

    evidenceIndex:
      consensus.evidenceIndex,

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
      consensus.segmentRange,

    segmentMedian:
      consensus.segmentMedian,

    calibrationActive:
      consensus
        .calibration
        .active
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
        ? 'V4.2 calibrated consensus ready ✓'
        : 'Limited evidence mode';


  setTimeout(
    () => {

      $('progress')
        .classList
        .add(
          'hidden'
        );

    },
    500
  );


  $('scan').disabled =
    false;


  /* =====================================================
     BENCHMARK
  ===================================================== */

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
   RENDER V4.2
========================================================= */

function renderV42({
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
    .remove(
      'hidden'
    );


  /* =====================================================
     MAIN SCORE

     This is now Evidence Index,
     not "% written by AI".
  ===================================================== */

  $('score')
    .textContent =
      `${consensus.evidenceIndex}%`;


  $('scaleFill')
    .style
    .width =
      `${consensus.evidenceIndex}%`;


  $('verdict')
    .textContent =
      consensus.verdict;


  const confidenceLabel =
    consensus.confidence >=
      75
      ? 'High'
      : consensus.confidence >=
          55
        ? 'Medium'
        : 'Low';


  $('confidence')
    .textContent =
      `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;


  /* =====================================================
     EXPLANATION
  ===================================================== */

  if (
    consensus.verdict ===
      'INCONCLUSIVE'
  ) {

    $('explain')
      .textContent =
`Evidence is not stable enough for a reliable AI/Human verdict.

Raw detector signal: ${consensus.rawSignal}%.

Reliability-adjusted evidence index: ${consensus.evidenceIndex}%.

Model disagreement: ${consensus.modelGap} points.

Segment deviation: ${consensus.segmentDeviation}.

Segment range: ${consensus.segmentRange} points.`;

  } else {

    $('explain')
      .textContent =
`AI Trace combined two detection engines, segment stability, and calibration safeguards.

TMR: ${tmrDocument}%.

ModernBERT: ${modernDocument}%.

Raw signal: ${consensus.rawSignal}%.

Reliability-adjusted index: ${consensus.evidenceIndex}%.`;
  }


  /* =====================================================
     TRACE DNA DISPLAY
  ===================================================== */

  const humanDisplay =
    100 -
    consensus.evidenceIndex;


  $('humanVal')
    .textContent =
      `${humanDisplay}%`;


  $('aiVal')
    .textContent =
      `${consensus.evidenceIndex}%`;


  $('uncertainVal')
    .textContent =
      `${consensus.uncertainty}%`;


  $('humanBar')
    .style
    .width =
      `${humanDisplay}%`;


  $('aiBar')
    .style
    .width =
      `${consensus.evidenceIndex}%`;


  $('uncertainBar')
    .style
    .width =
      `${consensus.uncertainty}%`;


  $('engineBadge')
    .textContent =
      (
        tmrWorked &&
        modernWorked
      )
        ? 'V4.2 • ROBUST CONSENSUS'
        : 'LIMITED EVIDENCE';


  /* =====================================================
     CALIBRATION STATUS
  ===================================================== */

  const calibrationText =
    consensus
      .calibration
      .active

      ? `Active — ${consensus.calibration.nearby} nearby benchmark samples, empirical AI rate ${consensus.calibration.localAIRate}%`

      : `Not active yet — ${consensus.calibration.samples}/${MIN_CALIBRATION_SAMPLES} known benchmark samples`;


  /* =====================================================
     EVIDENCE PANEL
  ===================================================== */

  const evidence = [

    [
      'TMR detector',

      tmrWorked
        ? `${tmrDocument}% raw AI signal`
        : 'Detector unavailable',

      tmrWorked
        ? 'Model A'
        : 'Unavailable'
    ],


    [
      'ModernBERT detector',

      modernWorked
        ? `${modernDocument}% raw AI signal`
        : 'Detector unavailable',

      modernWorked
        ? 'Model B'
        : 'Unavailable'
    ],


    [
      'Model agreement',

      `${consensus.modelAgreement}% agreement quality`,

      consensus.modelGap >=
        35
        ? 'High conflict'
        : 'Acceptable'
    ],


    [
      'Segment stability',

      `${consensus.segmentStability}% stability`,

      consensus
        .segmentsHighlyUnstable
        ? 'Unstable'
        : 'Acceptable'
    ],


    [
      'Segment median',

      `${consensus.segmentMedian}% AI signal`,

      'Robust signal'
    ],


    [
      'Calibration',

      calibrationText,

      consensus
        .calibration
        .active
        ? 'Active'
        : 'Learning'
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


  /* =====================================================
     METRICS
  ===================================================== */

  const metrics = {

    Words:
      profile.words,

    Sentences:
      profile.sentences,

    'Avg. words / sentence':
      profile
        .averageSentenceLength
        .toFixed(
          1
        ),

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

    'Raw detector signal':
      `${consensus.rawSignal}%`,

    'Evidence index':
      `${consensus.evidenceIndex}%`,

    'Evidence confidence':
      `${consensus.confidence}%`,

    'Model disagreement':
      `${consensus.modelGap} pts`,

    'Model agreement':
      `${consensus.modelAgreement}%`,

    'Segment stability':
      `${consensus.segmentStability}%`,

    'Segment deviation':
      consensus.segmentDeviation,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Segment median':
      `${consensus.segmentMedian}%`
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

  <span>
    ${key}
  </span>

  <b>
    ${value}
  </b>

</div>`
        )
        .join('');


  /* =====================================================
     TRACE MAP
  ===================================================== */

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


/* =========================================================
   BENCHMARK UTILITIES
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

    return safeParse(
      localStorage.getItem(
        SCAN_HISTORY_STORAGE
      ) || '[]',
      []
    );
  }
};
