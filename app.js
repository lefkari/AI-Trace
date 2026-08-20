/*
  ============================================================
  AI TRACE V5.4
  RELIABILITY-WEIGHTED CONSENSUS + OUTLIER DEFENSE

  Model A: TMR
  Model B: E5-small
  Model C: ModernBERT (conditional desktop judge)

  Features:
  - Local browser inference
  - Mobile-safe mode
  - 2/3 detector agreement
  - Outlier detector recognition
  - Reliability-weighted consensus
  - Segment confirmation
  - Human counter-evidence
  - Domain protection
  - Evidence-quality scoring
  - Safe abstention
  - Benchmark reliability learning
  - Zero paid API
  ============================================================
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';


env.allowLocalModels = false;
env.useBrowserCache = true;


/* ============================================================
   VERSION + MODELS
============================================================ */

const VERSION = '5.4';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';


const BENCH_KEY =
  'aiTraceBenchmarkV54';

const HISTORY_KEY =
  'aiTraceHistoryV54';


const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV53',
  'aiTraceBenchmarkV52',
  'aiTraceBenchmarkV51',
  'aiTraceBenchmarkV44',
  'aiTraceBenchmarkV43',
  'aiTraceBenchmarkV42',
  'aiTraceBenchmarkV41'
];


let tmr = null;
let e5 = null;
let modern = null;


/* ============================================================
   DOM
============================================================ */

const $ = id =>
  document.getElementById(id);

const textEl =
  $('text');


/* ============================================================
   DEVICE
============================================================ */

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
      .filter(Number.isFinite)
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
    return usable[middle];
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
    average(usable);

  return Math.sqrt(
    average(
      usable.map(
        value =>
          (value - mean) ** 2
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
      })[character]
  );
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


/* ============================================================
   UI
============================================================ */

function updateCount() {

  if (
    !$('count') ||
    !textEl
  ) {
    return;
  }

  $('count').textContent =
    `${wordCount(textEl.value)} words`;
}


function setProgress(
  percent,
  label
) {

  $('progress')
    ?.classList
    .remove('hidden');

  if ($('bar')) {
    $('bar').style.width =
      `${percent}%`;
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


function loadDemo() {

  if (!textEl) {
    return;
  }

  textEl.value = `Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

One major advantage of artificial intelligence is its ability to process large amounts of information quickly. Businesses can automate repetitive workflows, researchers can examine large datasets, and individuals can use intelligent tools to improve productivity. These systems can identify patterns that might be difficult for humans to notice manually.

At the same time, artificial intelligence introduces important challenges. Generated content may contain incorrect information, fabricated details, or biased conclusions. As AI-generated text becomes more natural, it can also become increasingly difficult to determine whether a document was written by a person or produced by a machine.

For this reason, future digital platforms may require stronger authenticity and provenance systems. Rather than relying on a single detection score, trustworthy tools should examine multiple signals, communicate uncertainty, and avoid presenting probabilistic evidence as absolute proof. Reliable AI detection will therefore depend on careful evaluation, transparent limitations, and continuous testing across many different types of content.`;

  updateCount();
}


textEl?.addEventListener(
  'input',
  updateCount
);


$('clear')?.addEventListener(
  'click',
  () => {

    if (textEl) {
      textEl.value = '';
    }

    updateCount();

    $('report')
      ?.classList
      .add('hidden');
  }
);


$('demo')?.addEventListener(
  'click',
  loadDemo
);


$('scan')?.addEventListener(
  'click',
  runSmartScan
);


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

  const sentenceLengths =
    sentences.map(
      sentence =>
        wordCount(sentence)
    );

  const paragraphLengths =
    paragraphs.map(
      paragraph =>
        wordCount(paragraph)
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
    punctuationPatterns.filter(
      regex =>
        (
          value.match(regex) ||
          []
        ).length > 0
    ).length;

  const quoteCount =
    (
      value.match(
        /["“”‘’]/g
      ) || []
    ).length;

  const dialogueLines =
    lines.filter(
      line =>
        /^[“"'—-]/.test(line) ||
        /[”"']$/.test(line)
    ).length;

  const firstPerson =
    (
      value.match(
        /\b(I|me|my|mine|we|us|our|ours)\b/gi
      ) || []
    ).length;

  const contractions =
    (
      value.match(
        /\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi
      ) || []
    ).length;

  const subjectiveMarkers =
    (
      value.match(
        /\b(I think|I believe|I suppose|I feel|in my view|perhaps|maybe|it seems to me|I do not know)\b/gi
      ) || []
    ).length;

  const transitions =
    (
      value.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result|on the other hand)\b/gi
      ) || []
    ).length;


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

    contractions,

    subjectiveMarkers,

    transitions
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

    academic:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract|research)\b/g
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
        /\b(aita|tldr|subreddit|upvote|downvote|throwaway|imo|lol)\b/g
      ),

    wiki:
      countMatches(
        content,
        /\b(was born|refers to|located in|population|history of|known for|founded|species)\b/g
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
        ? 5
        : 0,

    books:
      (
        profile.quoteCount >= 6 ||
        profile.dialogueLines >= 2
      )
        ? 5
        : 0
  };


  const sorted =
    Object.entries(signals)
      .sort(
        (a, b) =>
          b[1] - a[1]
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
          'low',

        score:
          1
      };
    }


    return {
      domain:
        'general',

      confidence:
        'low',

      score:
        0
    };
  }


  return {

    domain,

    confidence:
      score >= 5
        ? 'high'
        : score >= 3
          ? 'medium'
          : 'low',

    score
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
    0.68
  ) {

    score += 20;

    reasons.push(
      'high sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score += 12;

    reasons.push(
      'moderate sentence-length variation'
    );
  }


  if (
    profile.punctuationTypes >= 5
  ) {

    score += 12;

    reasons.push(
      'rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >= 3
  ) {

    score += 6;
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

    score += 10;

    reasons.push(
      'personal or subjective voice'
    );

  } else if (
    profile.firstPerson > 0 ||
    profile.subjectiveMarkers > 0
  ) {

    score += 4;
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

    score += 8;

    reasons.push(
      'irregular paragraph rhythm'
    );
  }


  if (
    profile.lexicalDiversity >= 0.65
  ) {

    score += 7;

  } else if (
    profile.lexicalDiversity >= 0.55
  ) {

    score += 4;
  }


  if (
    profile.transitions >= 5
  ) {

    score -= 7;
  }


  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {

    score += 10;

    reasons.push(
      'literary-domain protection'
    );
  }


  return {

    score:
      clamp(
        Math.round(score)
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

  if (tmr) {
    return tmr;
  }

  setState(
    'Loading TMR…'
  );

  setProgress(
    8,
    'Loading Model A…'
  );

  tmr =
    await pipeline(
      'text-classification',
      MODEL_TMR,
      {
        dtype:
          'q4f16'
      }
    );

  return tmr;
}


async function loadE5() {

  if (e5) {
    return e5;
  }

  setState(
    'Loading E5-small…'
  );

  setProgress(
    18,
    'Loading Model B…'
  );

  e5 =
    await pipeline(
      'text-classification',
      MODEL_E5,
      {
        dtype:
          'q4f16'
      }
    );

  return e5;
}


async function loadModern() {

  if (modern) {
    return modern;
  }

  setState(
    'Loading ModernBERT judge…'
  );

  setProgress(
    72,
    'Loading Model C…'
  );

  modern =
    await pipeline(
      'text-classification',
      MODEL_MODERN,
      {
        dtype:
          'q4f16'
      }
    );

  return modern;
}


/* ============================================================
   MODEL OUTPUT
============================================================ */

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
      ).toLowerCase();

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
    aiProbability(result) *
    100
  );
}


/* ============================================================
   THIRD MODEL ROUTING
============================================================ */

function shouldUseThirdModel({
  quickScores,
  segmentScores,
  domain,
  humanScore,
  words
}) {

  const modelGap =
    Math.abs(
      quickScores.tmr -
      quickScores.e5
    );


  const raw =
    median([
      quickScores.tmr,
      quickScores.e5
    ]);


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

    modelGap >= 15 ||

    (
      raw >= 35 &&
      raw <= 88
    ) ||

    segmentRange >= 40 ||

    segmentSD >= 18 ||

    humanScore >= 40 ||

    domain === 'books' ||

    domain === 'poetry'
  );
}


/* ============================================================
   JSON STORAGE
============================================================ */

function loadJSON(key) {

  try {

    return JSON.parse(
      localStorage.getItem(key) ||
      '[]'
    );

  } catch (error) {

    console.warn(
      `Could not read ${key}:`,
      error
    );

    return [];
  }
}


function saveJSON(
  key,
  data
) {

  try {

    localStorage.setItem(
      key,
      JSON.stringify(data)
    );

    return true;

  } catch (error) {

    console.warn(
      `Could not save ${key}:`,
      error
    );

    return false;
  }
}


/* ============================================================
   LEGACY BENCHMARK MIGRATION
============================================================ */

function normalizeLegacyRecord(record) {

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


  const scores =
    record.scores || {

      tmr:
        Number(
          record.models?.tmr ??
          record.tmr
        ),

      e5:
        Number(
          record.models?.e5 ??
          record.e5
        ),

      modern:
        Number(
          record.models?.modern ??
          record.modern
        )
    };


  const consensus =
    record.consensus || {

      raw:
        record.rawSignal ??
        record.raw ??
        50,

      calibrated:
        record.calibratedScore ??
        record.calibrated ??
        record.score ??
        50,

      quality:
        record.evidenceQuality ??
        record.quality ??
        record.confidence ??
        50,

      confidence:
        record.confidence ??
        50,

      uncertainty:
        record.uncertainty ??
        50,

      verdict:
        record.verdict ||
        'INCONCLUSIVE',

      modelSpread:
        record.modelGap ??
        record.modelSpread ??
        0,

      segmentRange:
        record.segmentRange ??
        0,

      segmentSD:
        record.segmentDeviation ??
        record.segmentSD ??
        0
    };


  return {

    ...record,

    truth,

    version:
      record.version ||
      record.appVersion ||
      'legacy',

    domain:
      record.domain ||
      'general',

    scores,

    consensus
  };
}


function loadBench() {

  let records =
    loadJSON(
      BENCH_KEY
    );


  if (
    Array.isArray(records) &&
    records.length
  ) {

    return records
      .map(
        normalizeLegacyRecord
      )
      .filter(Boolean);
  }


  for (
    const legacyKey
    of LEGACY_BENCH_KEYS
  ) {

    const legacy =
      loadJSON(
        legacyKey
      );


    if (
      Array.isArray(legacy) &&
      legacy.length
    ) {

      records =
        legacy
          .map(
            normalizeLegacyRecord
          )
          .filter(Boolean);


      saveJSON(
        BENCH_KEY,
        records
      );


      console.info(
        `Benchmark migrated from ${legacyKey}`
      );


      return records;
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
   BENCHMARK PREDICTIONS
============================================================ */

function verdictToPrediction(verdict) {

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


function prediction(record) {

  return verdictToPrediction(
    record?.consensus?.verdict ||
    record?.verdict ||
    ''
  );
}


/* ============================================================
   BENCHMARK METRICS
============================================================ */

function benchmarkMetrics(
  records = loadBench()
) {

  const rows =
    records.filter(
      record =>
        record.truth === 'AI' ||
        record.truth === 'HUMAN'
    );


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
      prediction(record);


    if (
      predicted ===
      'ABSTAIN'
    ) {

      if (
        record.truth === 'AI'
      ) {
        aiAbstain++;
      } else {
        humanAbstain++;
      }

      continue;
    }


    if (
      record.truth === 'AI' &&
      predicted === 'AI'
    ) {
      TP++;
    }


    if (
      record.truth === 'HUMAN' &&
      predicted === 'HUMAN'
    ) {
      TN++;
    }


    if (
      record.truth === 'HUMAN' &&
      predicted === 'AI'
    ) {
      FP++;
    }


    if (
      record.truth === 'AI' &&
      predicted === 'HUMAN'
    ) {
      FN++;
    }
  }


  const totalAI =
    rows.filter(
      record =>
        record.truth === 'AI'
    ).length;


  const totalHuman =
    rows.filter(
      record =>
        record.truth === 'HUMAN'
    ).length;


  const decided =
    TP +
    TN +
    FP +
    FN;


  const abstained =
    aiAbstain +
    humanAbstain;


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
    abstained,

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

    abstentionRate:
      percentage(
        abstained,
        rows.length
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
   DETECTOR METRICS
============================================================ */

function detectorPrediction(
  score,
  aiThreshold = 70,
  humanThreshold = 30
) {

  if (
    !Number.isFinite(score)
  ) {
    return 'ABSTAIN';
  }

  if (
    score >= aiThreshold
  ) {
    return 'AI';
  }

  if (
    score <= humanThreshold
  ) {
    return 'HUMAN';
  }

  return 'ABSTAIN';
}


function detectorMetrics(
  detectorName,
  records = loadBench()
) {

  const rows =
    records.filter(
      record =>
        (
          record.truth === 'AI' ||
          record.truth === 'HUMAN'
        ) &&
        Number.isFinite(
          Number(
            record.scores?.[
              detectorName
            ]
          )
        )
    );


  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;
  let abstain = 0;


  for (
    const record
    of rows
  ) {

    const score =
      Number(
        record.scores?.[
          detectorName
        ]
      );


    const predicted =
      detectorPrediction(
        score
      );


    if (
      predicted ===
      'ABSTAIN'
    ) {

      abstain++;
      continue;
    }


    if (
      record.truth === 'AI' &&
      predicted === 'AI'
    ) {
      TP++;
    }

    if (
      record.truth === 'HUMAN' &&
      predicted === 'HUMAN'
    ) {
      TN++;
    }

    if (
      record.truth === 'HUMAN' &&
      predicted === 'AI'
    ) {
      FP++;
    }

    if (
      record.truth === 'AI' &&
      predicted === 'HUMAN'
    ) {
      FN++;
    }
  }


  const totalAI =
    rows.filter(
      record =>
        record.truth === 'AI'
    ).length;


  const totalHuman =
    rows.filter(
      record =>
        record.truth === 'HUMAN'
    ).length;


  const decided =
    TP +
    TN +
    FP +
    FN;


  return {

    detector:
      detectorName,

    total:
      rows.length,

    TP,
    TN,
    FP,
    FN,

    abstain,

    coverage:
      percentage(
        decided,
        rows.length
      ),

    accuracy:
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
      )
  };
}


/* ============================================================
   BENCHMARK READINESS
============================================================ */

function calibrationReadiness(
  records = loadBench()
) {

  const binary =
    records.filter(
      record =>
        record.truth === 'AI' ||
        record.truth === 'HUMAN'
    );


  const ai =
    binary.filter(
      record =>
        record.truth === 'AI'
    ).length;


  const human =
    binary.filter(
      record =>
        record.truth === 'HUMAN'
    ).length;


  const domains =
    new Set(
      binary.map(
        record =>
          record.domain ||
          'general'
      )
    ).size;


  const minimumClass =
    Math.min(
      ai,
      human
    );


  let level =
    'NOT READY';


  if (
    minimumClass >= 100 &&
    binary.length >= 250 &&
    domains >= 5
  ) {

    level =
      'STRONG';

  } else if (
    minimumClass >= 50 &&
    binary.length >= 120 &&
    domains >= 4
  ) {

    level =
      'GOOD';

  } else if (
    minimumClass >= 20 &&
    binary.length >= 50 &&
    domains >= 3
  ) {

    level =
      'EARLY';

  } else if (
    minimumClass >= 10 &&
    binary.length >= 20
  ) {

    level =
      'EXPERIMENTAL';
  }


  return {

    level,

    total:
      binary.length,

    ai,
    human,
    domains,

    minimumClass,

    canInfluenceWeights:
      minimumClass >= 20 &&
      binary.length >= 50
  };
}


/* ============================================================
   DETECTOR RELIABILITY
============================================================ */

function detectorReliability(
  detectorName,
  records = loadBench()
) {

  const metrics =
    detectorMetrics(
      detectorName,
      records
    );


  /*
    Do NOT learn weight from tiny benchmark sets.
  */

  if (
    metrics.total < 20
  ) {

    return {

      ready:
        false,

      weight:
        1,

      metrics
    };
  }


  const coverage =
    metrics.coverage / 100;

  const accuracy =
    metrics.accuracy / 100;

  const fprPenalty =
    metrics.fpr / 100;

  const fnrPenalty =
    metrics.fnr / 100;


  let reliability =

    accuracy * 0.55 +

    coverage * 0.15 +

    (
      1 -
      fprPenalty
    ) * 0.20 +

    (
      1 -
      fnrPenalty
    ) * 0.10;


  reliability =
    clamp(
      reliability,
      0.25,
      1
    );


  return {

    ready:
      true,

    weight:
      reliability,

    metrics
  };
}


/* ============================================================
   BENCHMARK WEIGHTS
============================================================ */

function benchmarkWeights() {

  const records =
    loadBench();


  const readiness =
    calibrationReadiness(
      records
    );


  const defaults = {

    tmr:
      1,

    e5:
      1,

    modern:
      1,

    active:
      false,

    readiness
  };


  if (
    !readiness.canInfluenceWeights
  ) {
    return defaults;
  }


  const tmrReliability =
    detectorReliability(
      'tmr',
      records
    );

  const e5Reliability =
    detectorReliability(
      'e5',
      records
    );

  const modernReliability =
    detectorReliability(
      'modern',
      records
    );


  return {

    tmr:
      tmrReliability.ready
        ? tmrReliability.weight
        : 1,

    e5:
      e5Reliability.ready
        ? e5Reliability.weight
        : 1,

    modern:
      modernReliability.ready
        ? modernReliability.weight
        : 1,

    active:
      (
        tmrReliability.ready ||
        e5Reliability.ready ||
        modernReliability.ready
      ),

    readiness
  };
}


/* ============================================================
   WEIGHTED AVERAGE
============================================================ */

function weightedAverageScores(
  scores,
  weights
) {

  const entries = [

    [
      'tmr',
      scores.tmr
    ],

    [
      'e5',
      scores.e5
    ],

    [
      'modern',
      scores.modern
    ]

  ].filter(
    entry =>
      Number.isFinite(
        entry[1]
      )
  );


  if (
    !entries.length
  ) {
    return 50;
  }


  let numerator = 0;
  let denominator = 0;


  for (
    const [
      name,
      score
    ]
    of entries
  ) {

    const weight =
      Number(
        weights?.[
          name
        ]
      ) || 1;


    numerator +=
      score *
      weight;

    denominator +=
      weight;
  }


  return denominator
    ? numerator /
      denominator
    : 50;
}


/* ============================================================
   V5.4 OUTLIER DETECTION
============================================================ */

function detectModelOutlier(scores) {

  const entries = [

    {
      name:
        'tmr',
      score:
        scores.tmr
    },

    {
      name:
        'e5',
      score:
        scores.e5
    },

    {
      name:
        'modern',
      score:
        scores.modern
    }

  ].filter(
    item =>
      Number.isFinite(
        item.score
      )
  );


  if (
    entries.length < 3
  ) {

    return {

      found:
        false,

      name:
        null,

      score:
        null,

      pair:
        [],

      pairGap:
        null,

      outlierDistance:
        null
    };
  }


  /*
    Example:
    98, 54, 91

    Closest pair:
    98 and 91 = gap 7

    Remaining model:
    54
  */

  let bestPair = null;


  for (
    let i = 0;
    i < entries.length;
    i++
  ) {

    for (
      let j = i + 1;
      j < entries.length;
      j++
    ) {

      const gap =
        Math.abs(
          entries[i].score -
          entries[j].score
        );


      if (
        !bestPair ||
        gap < bestPair.gap
      ) {

        bestPair = {

          first:
            entries[i],

          second:
            entries[j],

          gap
        };
      }
    }
  }


  const remaining =
    entries.find(
      item =>
        item.name !==
          bestPair.first.name &&
        item.name !==
          bestPair.second.name
    );


  const pairMean =
    average([
      bestPair.first.score,
      bestPair.second.score
    ]);


  const distance =
    Math.abs(
      remaining.score -
      pairMean
    );


  /*
    Outlier only when:
    - two models agree reasonably well
    - third is far away
  */

  const found =
    bestPair.gap <= 16 &&
    distance >= 24;


  return {

    found,

    name:
      found
        ? remaining.name
        : null,

    score:
      found
        ? remaining.score
        : null,

    pair:
      found
        ? [
            bestPair.first.name,
            bestPair.second.name
          ]
        : [],

    pairScores:
      found
        ? [
            bestPair.first.score,
            bestPair.second.score
          ]
        : [],

    pairGap:
      Math.round(
        bestPair.gap
      ),

    pairMean:
      Math.round(
        pairMean
      ),

    outlierDistance:
      Math.round(
        distance
      )
  };
}


/* ============================================================
   TWO-OF-THREE AGREEMENT
============================================================ */

function analyzeAgreement(scores) {

  const entries = [

    {
      name:
        'tmr',
      score:
        scores.tmr
    },

    {
      name:
        'e5',
      score:
        scores.e5
    },

    {
      name:
        'modern',
      score:
        scores.modern
    }

  ].filter(
    item =>
      Number.isFinite(
        item.score
      )
  );


  const strongAI =
    entries.filter(
      item =>
        item.score >= 82
    );


  const ai =
    entries.filter(
      item =>
        item.score >= 70
    );


  const strongHuman =
    entries.filter(
      item =>
        item.score <= 18
    );


  const human =
    entries.filter(
      item =>
        item.score <= 30
    );


  return {

    strongAI:
      strongAI.length,

    ai:
      ai.length,

    strongHuman:
      strongHuman.length,

    human:
      human.length,

    twoStrongAI:
      strongAI.length >= 2,

    twoAI:
      ai.length >= 2,

    twoStrongHuman:
      strongHuman.length >= 2,

    twoHuman:
      human.length >= 2,

    total:
      entries.length
  };
}


/* ============================================================
   SEGMENT CONFIRMATION
============================================================ */

function analyzeSegments(
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

      median:
        50,

      range:
        0,

      sd:
        0,

      highAIShare:
        0,

      lowAIShare:
        0,

      strongAIConfirmation:
        false,

      strongHumanConfirmation:
        false
    };
  }


  const highAI =
    usable.filter(
      score =>
        score >= 80
    ).length;


  const lowAI =
    usable.filter(
      score =>
        score <= 25
    ).length;


  const highAIShare =
    highAI /
    usable.length;


  const lowAIShare =
    lowAI /
    usable.length;


  return {

    mean:
      Math.round(
        average(usable)
      ),

    median:
      Math.round(
        median(usable)
      ),

    range:
      Math.round(
        Math.max(...usable) -
        Math.min(...usable)
      ),

    sd:
      Math.round(
        standardDeviation(
          usable
        )
      ),

    highAIShare,

    lowAIShare,

    strongAIConfirmation:
      usable.length >= 2 &&
      highAIShare >= 0.75,

    strongHumanConfirmation:
      usable.length >= 2 &&
      lowAIShare >= 0.75
  };
}


/* ============================================================
   V5.4 CONSENSUS ENGINE
============================================================ */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  human,
  thirdUsed
}) {

  const activeEntries = [

    {
      name:
        'tmr',
      score:
        scores.tmr
    },

    {
      name:
        'e5',
      score:
        scores.e5
    },

    {
      name:
        'modern',
      score:
        scores.modern
    }

  ].filter(
    item =>
      Number.isFinite(
        item.score
      )
  );


  const active =
    activeEntries.map(
      item =>
        item.score
    );


  const weights =
    benchmarkWeights();


  const agreement =
    analyzeAgreement(
      scores
    );


  const outlier =
    detectModelOutlier(
      scores
    );


  const segmentAnalysis =
    analyzeSegments(
      segmentScores
    );


  const normalMedian =
    median(
      active
    );


  const normalWeighted =
    weightedAverageScores(
      scores,
      weights
    );


  let consensusScores =
    activeEntries.slice();


  /*
    V5.4 IMPORTANT:

    If 3 models exist and one is a clear outlier,
    we DO NOT delete it.

    We reduce its influence instead.
  */

  let effectiveWeights = {

    tmr:
      weights.tmr,

    e5:
      weights.e5,

    modern:
      weights.modern
  };


  if (
    outlier.found &&
    outlier.name
  ) {

    effectiveWeights[
      outlier.name
    ] *= 0.32;
  }


  /*
    Agreement bonus.

    If two models strongly agree in one direction,
    strengthen their relative influence.
  */

  if (
    agreement.twoStrongAI
  ) {

    for (
      const item
      of consensusScores
    ) {

      if (
        item.score >= 82
      ) {

        effectiveWeights[
          item.name
        ] *= 1.18;
      }
    }
  }


  if (
    agreement.twoStrongHuman
  ) {

    for (
      const item
      of consensusScores
    ) {

      if (
        item.score <= 18
      ) {

        effectiveWeights[
          item.name
        ] *= 1.18;
      }
    }
  }


  const reliabilityWeighted =
    weightedAverageScores(
      scores,
      effectiveWeights
    );


  /*
    Median remains useful against extreme values.

    Reliability weighted score gets more influence
    only when outlier or benchmark calibration exists.
  */

  let raw;


  if (
    outlier.found
  ) {

    raw =
      Math.round(
        normalMedian *
        0.35 +
        reliabilityWeighted *
        0.65
      );

  } else if (
    weights.active
  ) {

    raw =
      Math.round(
        normalMedian *
        0.50 +
        reliabilityWeighted *
        0.50
      );

  } else {

    raw =
      Math.round(
        normalMedian *
        0.65 +
        normalWeighted *
        0.35
      );
  }


  /*
    Segment confirmation may nudge raw consensus
    but never dominate document-level models.
  */

  if (
    segmentAnalysis.strongAIConfirmation &&
    agreement.twoAI
  ) {

    raw +=
      Math.round(
        (
          segmentAnalysis.mean -
          raw
        ) *
        0.10
      );
  }


  if (
    segmentAnalysis.strongHumanConfirmation &&
    agreement.twoHuman
  ) {

    raw +=
      Math.round(
        (
          segmentAnalysis.mean -
          raw
        ) *
        0.10
      );
  }


  raw =
    clamp(
      raw
    );


  const modelSpread =
    active.length > 1
      ? Math.max(...active) -
        Math.min(...active)
      : 100;


  const modelSD =
    Math.round(
      standardDeviation(
        active
      )
    );


  /*
    EFFECTIVE conflict is different from raw spread.

    Example:
    98 / 54 / 91

    Raw spread = 44

    But because 98 and 91 strongly agree and
    54 is an outlier, effective conflict should
    be much lower.
  */

  let effectiveModelConflict =
    modelSpread;


  if (
    outlier.found
  ) {

    effectiveModelConflict =
      Math.max(
        outlier.pairGap,
        Math.round(
          modelSpread *
          0.38
        )
      );
  }


  const segmentRange =
    segmentAnalysis.range;


  const segmentSD =
    segmentAnalysis.sd;


  let domainRisk = 6;


  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {

    domainRisk = 18;

  } else if (
    domain === 'academic'
  ) {

    domainRisk = 10;

  } else if (
    domain === 'general'
  ) {

    domainRisk = 7;
  }


  /*
    EVIDENCE QUALITY
  */

  let quality =
    100;


  quality -=
    Math.min(
      32,
      effectiveModelConflict *
      0.62
    );


  quality -=
    Math.min(
      26,
      segmentSD *
      0.55 +
      segmentRange *
      0.13
    );


  quality -=
    profile.words < 120
      ? 20
      : profile.words < 180
        ? 12
        : profile.words < 250
          ? 6
          : 0;


  quality -=
    language === 'English'
      ? 0
      : 35;


  quality -=
    domainRisk;


  quality -=
    thirdUsed
      ? 0
      : 6;


  if (
    active.length < 2
  ) {

    quality -= 40;
  }


  /*
    Positive evidence quality bonuses
  */

  if (
    agreement.twoStrongAI ||
    agreement.twoStrongHuman
  ) {

    quality += 8;
  }


  if (
    (
      agreement.twoStrongAI &&
      segmentAnalysis.strongAIConfirmation
    ) ||
    (
      agreement.twoStrongHuman &&
      segmentAnalysis.strongHumanConfirmation
    )
  ) {

    quality += 7;
  }


  if (
    outlier.found
  ) {

    /*
      Recognizing an outlier is better than treating
      the whole ensemble as chaotic.
    */

    quality += 4;
  }


  quality =
    clamp(
      Math.round(
        quality
      )
    );


  const instability =
    1 -
    quality / 100;


  /*
    Human evidence should counter AI scores,
    but cannot overpower strong 2/3 agreement
    when human evidence is weak.
  */

  let humanPenalty =
    human.score *
    (
      0.07 +
      instability *
      0.42
    ) *
    (
      raw / 100
    );


  if (
    agreement.twoStrongAI &&
    human.score < 30
  ) {

    humanPenalty *=
      0.55;
  }


  let calibrated =
    clamp(
      Math.round(
        raw -
        humanPenalty
      )
    );


  /*
    Literary false-positive protection
  */

  const literaryRisk =
    (
      domain === 'books' ||
      domain === 'poetry'
    );


  if (
    literaryRisk &&
    human.score >= 42 &&
    (
      effectiveModelConflict >= 18 ||
      segmentRange >= 38
    )
  ) {

    calibrated =
      Math.min(
        calibrated,
        64
      );
  }


  /*
    CONFLICT LEVEL
  */

  const severeConflict =
    (
      effectiveModelConflict >= 42
    ) ||
    (
      segmentRange >= 75
    ) ||
    (
      segmentSD >= 30
    );


  const highConflict =
    (
      effectiveModelConflict >= 28
    ) ||
    (
      segmentRange >= 62
    ) ||
    (
      segmentSD >= 25
    );


  /*
    However, confirmed 2/3 agreement plus
    segment confirmation can resolve an otherwise
    elevated raw spread.
  */

  const resolvedAIConflict =
    agreement.twoStrongAI &&
    segmentAnalysis.strongAIConfirmation &&
    human.score < 35 &&
    outlier.found;


  const resolvedHumanConflict =
    agreement.twoStrongHuman &&
    segmentAnalysis.strongHumanConfirmation &&
    human.score >= 35 &&
    outlier.found;


  const conflictResolved =
    resolvedAIConflict ||
    resolvedHumanConflict;


  /*
    UNCERTAINTY
  */

  let uncertainty =
    100 -
    quality;


  if (
    outlier.found &&
    !conflictResolved
  ) {

    uncertainty += 5;
  }


  if (
    conflictResolved
  ) {

    uncertainty -= 9;
  }


  if (
    raw >= 70 &&
    human.score >= 50
  ) {

    uncertainty +=
      (
        human.score -
        45
      ) *
      0.25;
  }


  uncertainty =
    clamp(
      Math.round(
        uncertainty
      ),
      5,
      95
    );


  const confidence =
    100 -
    uncertainty;


  /*
    FINAL VERDICT
  */

  let verdict =
    'INCONCLUSIVE';


  /*
    STRONG AI

    Either:
    A) normally strong consensus

    OR
    B) 2 strong models + strong segment confirmation
       + low human counter-evidence + detected outlier.
  */

  const normalStrongAI =
    (
      calibrated >= 86 &&
      quality >= 72 &&
      effectiveModelConflict < 26 &&
      segmentRange < 55 &&
      human.score < 45
    );


  const resolvedStrongAI =
    (
      calibrated >= 84 &&
      quality >= 66 &&
      agreement.twoStrongAI &&
      segmentAnalysis.strongAIConfirmation &&
      human.score < 35 &&
      (
        outlier.found ||
        effectiveModelConflict < 25
      )
    );


  if (
    language === 'English' &&
    active.length >= 2 &&
    (
      normalStrongAI ||
      resolvedStrongAI
    )
  ) {

    verdict =
      'Strong AI evidence';
  }


  /*
    LIKELY AI
  */

  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated >= 72 &&
    quality >= 57 &&
    human.score < 55 &&
    (
      agreement.twoAI ||
      effectiveModelConflict < 28
    )
  ) {

    verdict =
      'Likely AI';
  }


  /*
    STRONG HUMAN
  */

  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated <= 18 &&
    quality >= 58 &&
    human.score >= 45 &&
    (
      agreement.twoStrongHuman ||
      effectiveModelConflict < 25
    )
  ) {

    verdict =
      'Strong human evidence';
  }


  /*
    LIKELY HUMAN
  */

  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated <= 36 &&
    human.score >= 40 &&
    (
      agreement.twoHuman ||
      effectiveModelConflict < 30
    )
  ) {

    verdict =
      'Likely human';
  }


  /*
    HARD ABSTENTION GUARDS
  */

  if (
    language !== 'English' ||
    active.length < 2
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  /*
    Severe unresolved conflict still abstains.
  */

  if (
    severeConflict &&
    !conflictResolved
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  /*
    Literary domain protection.
  */

  if (
    verdict.includes('AI') &&
    literaryRisk &&
    human.score >= 42
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  /*
    Prevent AI verdict when models are pointing
    in genuinely opposite directions and there is
    no clear outlier resolution.
  */

  const oppositeDirections =
    active.some(
      score =>
        score >= 75
    ) &&
    active.some(
      score =>
        score <= 30
    );


  if (
    verdict.includes('AI') &&
    oppositeDirections &&
    !conflictResolved
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  return {

    raw,

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

    activeModels:
      active.length,

    normalMedian:
      Math.round(
        normalMedian
      ),

    normalWeighted:
      Math.round(
        normalWeighted
      ),

    reliabilityWeighted:
      Math.round(
        reliabilityWeighted
      ),

    modelSpread:
      Math.round(
        modelSpread
      ),

    effectiveModelConflict:
      Math.round(
        effectiveModelConflict
      ),

    modelSD,

    segmentRange,

    segmentSD,

    segmentMean:
      segmentAnalysis.mean,

    segmentMedian:
      segmentAnalysis.median,

    segmentAIConfirmation:
      segmentAnalysis.strongAIConfirmation,

    segmentHumanConfirmation:
      segmentAnalysis.strongHumanConfirmation,

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    thirdUsed,

    literaryRisk,

    severeConflict,

    highConflict,

    conflictResolved,

    agreement,

    outlier,

    benchmarkWeightsActive:
      weights.active,

    benchmarkReadiness:
      weights.readiness.level,

    weights: {

      tmr:
        Number(
          effectiveWeights
            .tmr
            .toFixed(3)
        ),

      e5:
        Number(
          effectiveWeights
            .e5
            .toFixed(3)
        ),

      modern:
        Number(
          effectiveWeights
            .modern
            .toFixed(3)
        )
    }
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
    wordCount(value);


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


    setState(
      'V5.4 Smart Scan running…'
    );


    const language =
      detectLanguage(value);


    const profile =
      createProfile(value);


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
      chunkText(value);


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


    /* ========================================================
       MODEL A
    ======================================================== */

    try {

      const modelA =
        await loadTMR();


      setProgress(
        22,
        'Running Model A — TMR…'
      );


      scores.tmr =
        await classify(
          modelA,
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
            27
          ),

          `Trace Map ${i + 1}/${chunks.length}…`
        );


        const score =
          await classify(
            modelA,
            chunks[i]
          );


        segmentScores.push(
          score
        );
      }

    } catch (error) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /* ========================================================
       MODEL B
    ======================================================== */

    try {

      const modelB =
        await loadE5();


      setProgress(
        60,
        'Running Model B — E5-small…'
      );


      scores.e5 =
        await classify(
          modelB,
          value
        );

    } catch (error) {

      console.error(
        'E5 failed:',
        error
      );
    }


    const mobile =
      isMobileDevice();


    /*
      MODEL C:
      desktop conditional judge.
    */

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


    /* ========================================================
       MODEL C
    ======================================================== */

    if (
      thirdUsed
    ) {

      try {

        const modelC =
          await loadModern();


        setProgress(
          76,
          'Running Model C — ModernBERT…'
        );


        scores.modern =
          await classify(
            modelC,
            value
          );

      } catch (error) {

        console.error(
          'ModernBERT failed:',
          error
        );

        thirdUsed =
          false;
      }
    }


    /*
      Safe fallback
    */

    if (
      !segmentScores.length
    ) {

      segmentScores =
        chunks.map(
          () => 50
        );
    }


    setProgress(
      88,
      'Detecting agreement and outliers…'
    );


    await new Promise(
      resolve =>
        requestAnimationFrame(
          resolve
        )
    );


    setProgress(
      92,
      'Calibrating reliability…'
    );


    const consensus =
      buildConsensus({

        scores,

        segmentScores,

        profile,

        language,

        domain:
          domainInfo.domain,

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

      domainScore:
        domainInfo.score,

      profile,

      scores,

      segmentScores,

      human,

      consensus
    };


    setProgress(
      96,
      'Building evidence report…'
    );


    render(scan);


    try {

      saveHistory(
        scan
      );

    } catch (error) {

      console.warn(
        'History save error:',
        error
      );
    }


    setProgress(
      100,
      'Trace complete'
    );


    if (
      mobile
    ) {

      setState(
        'V5.4 Mobile Safe Consensus ✓'
      );

    } else if (
      consensus.outlier.found
    ) {

      setState(
        `V5.4 Outlier Defense ✓ (${consensus.outlier.name})`
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V5.4 3-Model Consensus ✓'
      );

    } else {

      setState(
        'V5.4 Smart Consensus ✓'
      );
    }


    setTimeout(
      () => {

        try {

          benchmarkPrompt(
            scan
          );

        } catch (error) {

          console.warn(
            'Benchmark prompt error:',
            error
          );
        }

      },
      700
    );

  } catch (fatalError) {

    console.error(
      'AI TRACE V5.4 fatal error:',
      fatalError
    );


    setState(
      'Scan error'
    );


    alert(
`AI Trace could not complete the scan.

${fatalError?.message || 'Unknown error'}

Open the browser console for details.`
    );

  } finally {

    if ($('scan')) {

      $('scan').disabled =
        false;
    }


    hideProgress();
  }
}


/* ============================================================
   RENDER
============================================================ */

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


  if ($('score')) {

    $('score').textContent =
      `${consensus.calibrated}%`;
  }


  if ($('scaleFill')) {

    $('scaleFill').style.width =
      `${consensus.calibrated}%`;
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

    let explanation =
      `Raw consensus ${consensus.raw}%. Reliability-calibrated AI signal ${consensus.calibrated}%. Evidence quality ${consensus.quality}%.`;


    if (
      consensus.outlier.found
    ) {

      explanation +=
        ` ${consensus.outlier.name.toUpperCase()} was identified as a possible detector outlier because the other two models were only ${consensus.outlier.pairGap} points apart while the outlier was ${consensus.outlier.outlierDistance} points from their mean.`;
    }


    if (
      consensus.conflictResolved
    ) {

      explanation +=
        ' The model conflict was partially resolved by strong multi-model agreement and matching segment evidence.';
    }


    if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {

      explanation +=
        ' AI Trace abstained because the remaining evidence was not reliable enough for a confident classification.';
    }


    $('explain').textContent =
      explanation;
  }


  /* ========================================================
     TRACE DNA
  ======================================================== */

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


  if ($('humanVal')) {

    $('humanVal').textContent =
      `${humanDisplay}%`;
  }


  if ($('aiVal')) {

    $('aiVal').textContent =
      `${consensus.calibrated}%`;
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
      `${consensus.calibrated}%`;
  }


  if ($('uncertainBar')) {

    $('uncertainBar').style.width =
      `${consensus.uncertainty}%`;
  }


  if ($('engineBadge')) {

    if (
      consensus.outlier.found
    ) {

      $('engineBadge').textContent =
        'V5.4 • OUTLIER DEFENSE';

    } else if (
      consensus.benchmarkWeightsActive
    ) {

      $('engineBadge').textContent =
        'V5.4 • RELIABILITY CALIBRATED';

    } else if (
      consensus.thirdUsed
    ) {

      $('engineBadge').textContent =
        'V5.4 • 3-MODEL CONSENSUS';

    } else if (
      isMobileDevice()
    ) {

      $('engineBadge').textContent =
        'V5.4 • MOBILE SAFE';

    } else {

      $('engineBadge').textContent =
        'V5.4 • SMART CONSENSUS';
    }
  }


  const humanReasons =
    human.reasons.length
      ? human.reasons
          .slice(
            0,
            5
          )
          .join(' • ')
      : 'No strong human-style counter-signals';


  const modernText =
    Number.isFinite(
      scores.modern
    )
      ? `${scores.modern}% AI signal`
      : isMobileDevice()
        ? 'Disabled on mobile for memory stability'
        : 'Not required by Smart Scan';


  let outlierText =
    'No clear detector outlier';


  if (
    consensus.outlier.found
  ) {

    outlierText =
      `${consensus.outlier.name.toUpperCase()} (${consensus.outlier.score}%) flagged as possible outlier. Agreement pair: ${consensus.outlier.pair.join(' + ')} at ${consensus.outlier.pairScores.join('% / ')}%.`;
  }


  let agreementText =
    `${consensus.agreement.ai}/${consensus.agreement.total} models ≥70% AI`;


  if (
    consensus.agreement.twoStrongAI
  ) {

    agreementText =
      `${consensus.agreement.strongAI}/${consensus.agreement.total} models show strong AI evidence`;

  } else if (
    consensus.agreement.twoStrongHuman
  ) {

    agreementText =
      `${consensus.agreement.strongHuman}/${consensus.agreement.total} models show strong human evidence`;
  }


  const evidence = [

    [
      'Final decision',
      consensus.verdict,
      'Outcome'
    ],

    [
      'Reliability-calibrated AI signal',
      `${consensus.calibrated}%`,
      'Primary'
    ],

    [
      'Raw consensus',
      `${consensus.raw}%`,
      'Diagnostic'
    ],

    [
      'Reliability-weighted signal',
      `${consensus.reliabilityWeighted}%`,
      'Consensus'
    ],

    [
      'Model agreement',
      agreementText,
      consensus.agreement.twoStrongAI ||
      consensus.agreement.twoStrongHuman
        ? 'Strong'
        : 'Mixed'
    ],

    [
      'Outlier analysis',
      outlierText,
      consensus.outlier.found
        ? 'Outlier detected'
        : 'Clear'
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
      'TMR detector',
      Number.isFinite(
        scores.tmr
      )
        ? `${scores.tmr}% AI signal`
        : 'Unavailable',
      'Model A'
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
      `${consensus.modelSpread} points`,
      consensus.modelSpread >= 40
        ? 'Large'
        : 'Acceptable'
    ],

    [
      'Effective conflict',
      `${consensus.effectiveModelConflict} points`,
      consensus.conflictResolved
        ? 'Resolved'
        : consensus.severeConflict
          ? 'Critical'
          : consensus.highConflict
            ? 'Elevated'
            : 'Acceptable'
    ],

    [
      'Segment confirmation',
      `Mean ${consensus.segmentMean}% · range ${consensus.segmentRange} points`,
      consensus.segmentAIConfirmation
        ? 'AI confirmed'
        : consensus.segmentHumanConfirmation
          ? 'Human confirmed'
          : 'Mixed'
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      consensus.literaryRisk
        ? 'Protected domain'
        : 'Routing'
    ],

    [
      'Benchmark calibration',
      consensus.benchmarkWeightsActive
        ? `${consensus.benchmarkReadiness} — active`
        : `${consensus.benchmarkReadiness} — collecting samples`,
      consensus.benchmarkWeightsActive
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

  <small>
    ${escapeHTML(item[1])}
  </small>

</div>

`
        )
        .join('');
  }


  /* ========================================================
     METRICS
  ======================================================== */

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

    'Raw consensus':
      `${consensus.raw}%`,

    'Weighted consensus':
      `${consensus.reliabilityWeighted}%`,

    'Calibrated signal':
      `${consensus.calibrated}%`,

    'Model spread':
      `${consensus.modelSpread} pts`,

    'Effective conflict':
      `${consensus.effectiveModelConflict} pts`,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Evidence quality':
      `${consensus.quality}%`,

    'Outlier':
      consensus.outlier.found
        ? consensus.outlier.name
        : 'None',

    '2/3 AI agreement':
      consensus.agreement.twoAI
        ? 'Yes'
        : 'No',

    'Strong segment AI':
      consensus.segmentAIConfirmation
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

  <span>
    ${escapeHTML(key)}
  </span>

  <b>
    ${escapeHTML(String(value))}
  </b>

</div>

`
        )
        .join('');
  }


  /* ========================================================
     TRACE MAP
  ======================================================== */

  const chunks =
    chunkText(
      textEl.value.trim()
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

    <b>
      Segment ${index + 1}
    </b>

    <span>
      ${score}% TMR diagnostic
    </span>

  </div>

  <div class="segmentMeter">

    <i style="width:${score}%"></i>

  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        320
      )
    )}${
      chunk.length > 320
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


  renderBenchmarkPanel();
}


/* ============================================================
   BENCHMARK PROMPT
============================================================ */

function benchmarkPrompt(scan) {

  const answer =
    prompt(
`AI TRACE V5.4 BENCHMARK

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
        record.truth === truth
    ).length;


  const record = {

    id:
      `${prefix}-${String(
        sameTruthCount + 1
      ).padStart(
        3,
        '0'
      )}`,

    truth,

    source,

    savedAt:
      new Date()
        .toISOString(),

    ...scan
  };


  records.push(
    record
  );


  saveBench(
    records
  );


  renderBenchmarkPanel();


  const results =
    benchmarkMetrics(
      records
    );


  const readiness =
    calibrationReadiness(
      records
    );


  alert(
`Benchmark saved: ${record.id}

Known AI: ${results.totalAI}
Known HUMAN: ${results.totalHuman}

Coverage: ${results.coverage}%
Selective accuracy: ${results.selectiveAccuracy}%

False-positive rate: ${results.fpr}%
False-negative rate: ${results.fnr}%

AI abstention rate: ${results.aiAbstainRate}%
Human abstention rate: ${results.humanAbstainRate}%

Calibration status: ${readiness.level}

Development measurements only.`
  );
}


/* ============================================================
   DOMAIN METRICS
============================================================ */

function domainMetrics(
  records = loadBench()
) {

  const domains =
    new Map();


  for (
    const record
    of records
  ) {

    if (
      record.truth !== 'AI' &&
      record.truth !== 'HUMAN'
    ) {
      continue;
    }


    const domain =
      record.domain ||
      'general';


    if (
      !domains.has(domain)
    ) {

      domains.set(
        domain,
        []
      );
    }


    domains
      .get(domain)
      .push(record);
  }


  return [
    ...domains.entries()
  ]
    .map(
      (
        [
          domain,
          rows
        ]
      ) => {

        const metrics =
          benchmarkMetrics(
            rows
          );


        return {

          domain,

          samples:
            metrics.total,

          coverage:
            metrics.coverage,

          accuracy:
            metrics.selectiveAccuracy,

          fpr:
            metrics.fpr,

          fnr:
            metrics.fnr,

          abstention:
            metrics.abstentionRate
        };
      }
    )
    .sort(
      (a, b) =>
        b.samples -
        a.samples
    );
}


/* ============================================================
   BENCHMARK DASHBOARD
============================================================ */

function renderBenchmarkPanel() {

  const report =
    $('report');


  if (!report) {
    return;
  }


  let panel =
    $('benchmarkPanelV5');


  if (!panel) {

    panel =
      document.createElement(
        'section'
      );


    panel.id =
      'benchmarkPanelV5';


    panel.className =
      'panel devPanel';


    report.appendChild(
      panel
    );
  }


  const records =
    loadBench();


  const results =
    benchmarkMetrics(
      records
    );


  const readiness =
    calibrationReadiness(
      records
    );


  const weights =
    benchmarkWeights();


  const tmrStats =
    detectorMetrics(
      'tmr',
      records
    );


  const e5Stats =
    detectorMetrics(
      'e5',
      records
    );


  const modernStats =
    detectorMetrics(
      'modern',
      records
    );


  const detectorHTML = [

    [
      'TMR',
      tmrStats,
      weights.tmr
    ],

    [
      'E5-small',
      e5Stats,
      weights.e5
    ],

    [
      'ModernBERT',
      modernStats,
      weights.modern
    ]

  ]
    .map(
      (
        [
          name,
          stats,
          weight
        ]
      ) => `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(name)}
    </span>

    <span>
      ${stats.total} samples
    </span>

  </div>

  <small>
    Accuracy ${stats.accuracy}%
    · Coverage ${stats.coverage}%
    · FPR ${stats.fpr}%
    · FNR ${stats.fnr}%
    · Reliability weight ${Number(weight).toFixed(2)}
  </small>

</div>

`
    )
    .join('');


  const domains =
    domainMetrics(
      records
    );


  const domainHTML =
    domains
      .map(
        item => `

<div class="metric">

  <span>
    ${escapeHTML(item.domain)}
  </span>

  <b>
    ${item.samples} samples
    · Acc ${item.accuracy}%
    · FPR ${item.fpr}%
    · Abstain ${item.abstention}%
  </b>

</div>

`
      )
      .join('');


  const recent =
    records
      .slice()
      .reverse()
      .slice(
        0,
        12
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
    ${escapeHTML(
      record.domain ||
      'general'
    )}
    · result ${escapeHTML(
      String(
        record.consensus
          ?.calibrated ??
        '?'
      )
    )}%
    · quality ${escapeHTML(
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

`
      )
      .join('');


  panel.innerHTML = `

<span class="over">
  V5.4 BENCHMARK • DEVELOPMENT ONLY
</span>

<h2>
  Reliability Intelligence
</h2>

<p class="sub">
  Known AI and HUMAN samples measure detector reliability. Inconclusive results remain abstentions rather than being silently counted as correct.
</p>


<h3>
  Calibration status
</h3>

<div class="evidence">

  <div class="ev">

    <div class="evTop">

      <span>
        ${readiness.level}
      </span>

      <span>
        ${
          readiness.canInfluenceWeights
            ? 'RELIABILITY LEARNING ACTIVE'
            : 'COLLECTING DATA'
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

</div>


<h3>
  Ensemble performance
</h3>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${results.total}</b>
  </div>

  <div class="metric">
    <span>Decided samples</span>
    <b>${results.decided}</b>
  </div>

  <div class="metric">
    <span>Abstentions</span>
    <b>${results.abstained}</b>
  </div>

  <div class="metric">
    <span>Coverage</span>
    <b>${results.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${results.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${results.precision}%</b>
  </div>

  <div class="metric">
    <span>AI recall</span>
    <b>${results.recall}%</b>
  </div>

  <div class="metric">
    <span>Human specificity</span>
    <b>${results.specificity}%</b>
  </div>

  <div class="metric">
    <span>False-positive rate</span>
    <b>${results.fpr}%</b>
  </div>

  <div class="metric">
    <span>False-negative rate</span>
    <b>${results.fnr}%</b>
  </div>

  <div class="metric">
    <span>AI abstention rate</span>
    <b>${results.aiAbstainRate}%</b>
  </div>

  <div class="metric">
    <span>Human abstention rate</span>
    <b>${results.humanAbstainRate}%</b>
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
      <span>No domain data yet</span>
      <b>—</b>
    </div>
    `
  }

</div>


<h3>
  Recent benchmark records
</h3>

<div class="evidence">

  ${
    recent ||
    `
    <div class="ev">
      <small>
        No V5.4 benchmark records yet.
      </small>
    </div>
    `
  }

</div>

`;
}


/* ============================================================
   HISTORY
============================================================ */

function saveHistory(scan) {

  try {

    const history =
      loadJSON(
        HISTORY_KEY
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

  } catch (error) {

    console.warn(
      'History error:',
      error
    );
  }
}


/* ============================================================
   DEVELOPER UTILITIES
============================================================ */

window.AITraceV54 = {


  benchmark() {

    const records =
      loadBench();


    return {

      version:
        VERSION,

      readiness:
        calibrationReadiness(
          records
        ),

      ensemble:
        benchmarkMetrics(
          records
        ),

      detectors: {

        tmr:
          detectorMetrics(
            'tmr',
            records
          ),

        e5:
          detectorMetrics(
            'e5',
            records
          ),

        modern:
          detectorMetrics(
            'modern',
            records
          )
      },

      domains:
        domainMetrics(
          records
        ),

      weights:
        benchmarkWeights(),

      records
    };
  },


  history() {

    return loadJSON(
      HISTORY_KEY
    );
  },


  exportBenchmark() {

    const report =
      this.benchmark();


    const blob =
      new Blob(
        [
          JSON.stringify(
            report,
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
      `AI-Trace-V54-Benchmark-${Date.now()}.json`;


    document.body
      .appendChild(
        anchor
      );


    anchor.click();


    anchor.remove();


    URL.revokeObjectURL(
      url
    );
  },


  clearBenchmark() {

    const confirmation =
      confirm(
        'Delete all V5.4 benchmark records from this device?'
      );


    if (!confirmation) {
      return;
    }


    localStorage.removeItem(
      BENCH_KEY
    );


    renderBenchmarkPanel();


    alert(
      'V5.4 benchmark data deleted.'
    );
  },


  clearHistory() {

    const confirmation =
      confirm(
        'Delete local V5.4 scan history?'
      );


    if (!confirmation) {
      return;
    }


    localStorage.removeItem(
      HISTORY_KEY
    );


    alert(
      'V5.4 scan history deleted.'
    );
  }
};


/* ============================================================
   INIT
============================================================ */

updateCount();


setTimeout(
  renderBenchmarkPanel,
  350
);


console.info(
  `AI TRACE V${VERSION} loaded`
);
