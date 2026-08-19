/*
============================================================
 AI TRACE V4.3
 Benchmark + Calibration Research Engine

 ONE FILE BUILD
 - TMR detector
 - ModernBERT detector
 - 2-model consensus
 - segment stability
 - evidence quality
 - calibrated detection score
 - HUMAN / AI / MIXED benchmark labels
 - confusion matrix
 - false positive rate
 - false negative rate
 - precision / recall / specificity
 - local benchmark storage
 - experimental threshold search
============================================================
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

const VERSION = '4.3';

const TMR_MODEL =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODERN_MODEL =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCHMARK_KEY =
  'aiTraceBenchmarkV43';

const HISTORY_KEY =
  'aiTraceHistoryV43';

let tmrClassifier = null;
let modernClassifier = null;


/* =========================================================
   DOM
========================================================= */

const $ = id =>
  document.getElementById(id);

const text = $('text');


/* =========================================================
   INPUT
========================================================= */

text.oninput = () => {

  const words =
    text.value.trim()
      ? text.value
          .trim()
          .split(/\s+/)
          .filter(Boolean)
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


$('scan').onclick =
  run;


/* =========================================================
   UI PROGRESS
========================================================= */

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
    0.8
  )
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

  const cleanWords =
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
      cleanWords
    ).size /
    Math.max(
      1,
      cleanWords.length
    );

  const sentenceLengths =
    sentences.map(
      sentence =>
        sentence
          .split(/\s+/)
          .filter(Boolean)
          .length
    );

  const averageLength =
    mean(
      sentenceLengths
    );

  const variance =
    sentenceLengths.length
      ? mean(
          sentenceLengths.map(
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


/* =========================================================
   TEXT SEGMENTATION
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


/* =========================================================
   MODELS
========================================================= */

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
    'Loading detector B…'
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
   CLASSIFIER OUTPUT
========================================================= */

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
      label === 'label_1'
    ) {

      ai =
        Math.max(
          ai ?? 0,
          score
        );
    }

    if (
      label === 'human' ||
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
  model,
  value
) {

  const result =
    await model(
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


/* =========================================================
   RAW MODEL CONSENSUS
========================================================= */

function rawConsensus(
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
  tmr,
  modern,
  segments,
  language,
  words,
  bothModels
}) {

  /*
    Starts at 100 and loses confidence
    when evidence becomes unstable.
  */

  let quality =
    100;

  if (
    !bothModels
  ) {

    quality -=
      40;
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
    segmentRange >=
      70
  ) {

    quality -=
      15;
  }

  if (
    words <
      150
  ) {

    quality -=
      10;
  }

  if (
    language !==
      'English'
  ) {

    quality -=
      25;
  }

  return clamp(
    Math.round(
      quality
    ),
    5,
    100
  );
}


/* =========================================================
   CALIBRATED DETECTION SCORE

   Pulls uncertain raw detector output
   toward neutral 50 rather than pretending
   a noisy model score is a probability.
========================================================= */

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


/* =========================================================
   VERDICT
========================================================= */

function verdictFor({
  calibratedScore,
  evidenceQuality,
  segmentRange,
  modelGap
}) {

  /*
    Hard safety gates.
  */

  if (
    evidenceQuality <
      55
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    segmentRange >=
      70
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    modelGap >=
      35
  ) {

    return 'INCONCLUSIVE';
  }

  if (
    calibratedScore >=
      85
  ) {

    return 'Strong AI evidence';
  }

  if (
    calibratedScore >=
      68
  ) {

    return 'Likely AI';
  }

  if (
    calibratedScore <=
      15
  ) {

    return 'Strong human evidence';
  }

  if (
    calibratedScore <=
      32
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
  records
) {

  localStorage.setItem(
    BENCHMARK_KEY,
    JSON.stringify(
      records
    )
  );
}


/* =========================================================
   BENCHMARK IDS
========================================================= */

function nextID(
  truth,
  records
) {

  let prefix;

  if (
    truth === 'AI'
  ) {

    prefix = 'A';

  } else if (
    truth === 'HUMAN'
  ) {

    prefix = 'H';

  } else {

    prefix = 'M';
  }

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
   BENCHMARK PREDICTION
========================================================= */

function benchmarkPrediction(
  record,
  aiThreshold = 68,
  humanThreshold = 32
) {

  if (
    record.evidenceQuality <
      55 ||
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


/* =========================================================
   BENCHMARK METRICS
========================================================= */

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
        accuracy * 100
      ),

    precision:
      Math.round(
        precision * 100
      ),

    recall:
      Math.round(
        recall * 100
      ),

    specificity:
      Math.round(
        specificity * 100
      ),

    falsePositiveRate:
      Math.round(
        fpr * 100
      ),

    falseNegativeRate:
      Math.round(
        fnr * 100
      )
  };
}


/* =========================================================
   EXPERIMENTAL THRESHOLD SEARCH

   Does NOT affect production verdict yet.
   We only use this after collecting enough
   HUMAN + AI benchmark samples.
========================================================= */

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

      /*
        We strongly penalize
        false positives.
      */

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


/* =========================================================
   STORE BENCHMARK
========================================================= */

function recordBenchmark(
  data
) {

  const answer =
    prompt(
`AI TRACE V4.3 BENCHMARK

Only label this if you KNOW the true origin.

AI
HUMAN
MIXED

Cancel / empty = skip`
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
      'Source / note:',
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

  const best =
    searchThresholds(
      records
    );

  let message =
`Saved ${id}

KNOWN BINARY SAMPLES: ${metrics.samples}

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
False Negative Rate: ${metrics.falseNegativeRate}%`;

  if (
    best
  ) {

    message +=

`\n\nEXPERIMENTAL THRESHOLDS

Human <= ${best.humanThreshold}%
AI >= ${best.aiThreshold}%

These thresholds are research-only until the benchmark is much larger.`;
  }

  alert(
    message
  );
}


/* =========================================================
   HISTORY
========================================================= */

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


/* =========================================================
   MAIN ANALYSIS
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
      80
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


  /* =======================================================
     TMR DOCUMENT + SEGMENTS
  ======================================================= */

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


  /* =======================================================
     MODERNBERT
  ======================================================= */

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


  /* =======================================================
     CONSENSUS + CALIBRATION
  ======================================================= */

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
      rawConsensus(
        tmrScore,
        modernScore
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


  /* =======================================================
     RENDER
  ======================================================= */

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


  /* =======================================================
     HISTORY
  ======================================================= */

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


  /* =======================================================
     ASK GROUND TRUTH
  ======================================================= */

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
   RENDER
========================================================= */

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


  /* =======================================================
     MAIN SCORE
  ======================================================= */

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


  /* =======================================================
     TRACE DNA
  ======================================================= */

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
      'V4.3 • BENCHMARK CALIBRATION';


  /* =======================================================
     EVIDENCE
  ======================================================= */

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


  /* =======================================================
     DOCUMENT PROFILE
  ======================================================= */

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


  /* =======================================================
     TRACE MAP
  ======================================================= */

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

      behavior:
        'smooth',

      block:
        'start'
    });
}


/* =========================================================
   DEVELOPER BENCHMARK UTILITIES
========================================================= */

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


  exportBenchmark() {

    const data =
      JSON.stringify(
        this.benchmark(),
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
      `AI-Trace-V43-Benchmark-${Date.now()}.json`;

    anchor.click();

    URL.revokeObjectURL(
      url
    );
  },


  clearBenchmark() {

    if (
      !confirm(
        'Delete all V4.3 benchmark data?'
      )
    ) {

      return;
    }

    localStorage.removeItem(
      BENCHMARK_KEY
    );

    alert(
      'Benchmark cleared.'
    );
  }
};
