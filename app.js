/* =========================================================
   AI TRACE V5.3
   BENCHMARK-FIRST CALIBRATION ENGINE
========================================================= */

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;


/* =========================================================
   CONFIG
========================================================= */

const VERSION = '5.3';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCH_KEY =
  'aiTraceBenchmarkV53';

const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV52',
  'aiTraceBenchmarkV51',
  'aiTraceBenchmarkV5'
];

const HISTORY_KEY =
  'aiTraceHistoryV53';


/* =========================================================
   MODEL INSTANCES
========================================================= */

let tmr = null;
let e5 = null;
let modern = null;


/* =========================================================
   DOM
========================================================= */

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
   WORD COUNTER
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


/* =========================================================
   DEMO
========================================================= */

function loadDemo() {

  if (!textEl) {
    return;
  }

  textEl.value = `Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

One of the most important advantages of artificial intelligence is its ability to process large amounts of information quickly. Organizations can use AI-powered tools to identify patterns, automate repetitive processes, and support better decision-making. For example, businesses can analyze customer behavior, doctors can receive assistance when examining medical information, and researchers can process datasets that would be extremely difficult to evaluate manually.

However, the growing use of artificial intelligence also creates important challenges. AI-generated information can sometimes be inaccurate, misleading, or difficult to distinguish from content created by humans. Synthetic images, artificial voices, and automatically generated articles are becoming increasingly realistic.

The future will therefore require more than simply developing increasingly powerful artificial intelligence systems. Society will also need technologies that provide transparency, verification, and evidence about how digital content was created or modified.`;

  updateCount();
}


/* =========================================================
   PROGRESS
========================================================= */

function setProgress(
  percent,
  label
) {

  $('progress')
    ?.classList
    .remove('hidden');

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
    450
  );
}


function setState(label) {

  if ($('modelState')) {

    $('modelState').textContent =
      label;
  }
}


/* =========================================================
   BASIC HELPERS
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
      (a, b) => a + b,
      0
    ) /
    usable.length
  );
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


function median(values) {

  const usable =
    values
      .filter(Number.isFinite)
      .slice()
      .sort(
        (a, b) => a - b
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


function wordCount(value) {

  const trimmed =
    String(
      value || ''
    ).trim();

  if (!trimmed) {
    return 0;
  }

  return trimmed
    .split(/\s+/)
    .filter(Boolean)
    .length;
}


function escapeHTML(value) {

  return String(value)
    .replace(
      /[&<>"']/g,
      character => ({
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
    String(value)
      .match(regex) ||
    []
  ).length;
}


function percentage(
  numerator,
  denominator
) {

  return denominator
    ? Math.round(
        numerator /
        denominator *
        100
      )
    : 0;
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
    latin / total > 0.82
  )
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
      wordCount
    );


  const paragraphLengths =
    paragraphs.map(
      wordCount
    );


  const cleanWords =
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
            value.match(regex) ||
            []
          ).length > 0
      )
      .length;


  const quoteCount =
    countMatches(
      value,
      /["“”‘’]/g
    );


  const semicolonCount =
    countMatches(
      value,
      /;/g
    );


  const dialogueLines =
    lines.filter(
      line =>
        /^[“"'—-]/.test(line) ||
        /[”"']$/.test(line)
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
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result)\b/gi
    );


  const titleReferences =
    countMatches(
      value,
      /\b(Mr|Mrs|Miss|Ms|Sir|Lady|Lord|Dr)\.?\s+[A-Z][a-z]+/g
    );


  const narrativeMarkers =
    countMatches(
      value,
      /\b(she|he|her|his|hers|him|father|mother|daughter|son|sister|brother|friend|marriage|family|house|young|years|woman|man)\b/gi
    );


  const literaryVocabulary =
    countMatches(
      value,
      /\b(affection|disposition|esteem|governess|intimacy|temper|authority|misfortune|existence|indulgent|remembrance|caresses|enjoyments|judgment|vex|distress|attached|acquaintance|gentleman|lady)\b/gi
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
        cleanWords
      ).size /
      Math.max(
        1,
        cleanWords.length
      ),

    punctuationTypes,

    quoteCount,

    semicolonCount,

    dialogueLines,

    firstPerson,

    contractions,

    transitions,

    titleReferences,

    narrativeMarkers,

    literaryVocabulary
  };
}


/* =========================================================
   DOMAIN DETECTION
========================================================= */

function estimateDomain(
  value,
  profile
) {

  const content =
    value.toLowerCase();


  const scores = {

    academic:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract|research|findings|methodology|sample)\b/g
      ) * 2,


    recipes:
      countMatches(
        content,
        /\b(cup|cups|tablespoon|teaspoon|ingredients|preheat|oven|bake|stir|chop|minutes|serve|flour|sugar)\b/g
      ) * 2,


    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|experience|bought|customer)\b/g
      ) * 2,


    reddit:
      countMatches(
        content,
        /\b(aita|tldr|subreddit|upvote|downvote|edit:|throwaway|imo|lol|op)\b/g
      ) * 3,


    encyclopedia:
      countMatches(
        content,
        /\b(was born|refers to|located in|population|history of|known for|founded|species|geography|is a type of)\b/g
      ) * 2,


    news:
      countMatches(
        content,
        /\b(reuters|reported|according to|officials|government|minister|president|announced|agency|spokesperson|statement)\b/g
      ) * 2,


    poetry:
      0,


    books:
      0
  };


  if (
    profile.lineBreaks >= 5 &&
    profile.averageLineLength < 70
  ) {

    scores.poetry += 8;
  }


  scores.books +=
    profile.titleReferences * 3;


  scores.books +=
    Math.min(
      12,
      profile.literaryVocabulary * 2
    );


  scores.books +=
    Math.min(
      8,
      Math.floor(
        profile.narrativeMarkers / 3
      )
    );


  if (
    profile.averageSentenceLength >= 25
  ) {

    scores.books += 3;
  }


  if (
    profile.semicolonCount >= 2
  ) {

    scores.books += 3;
  }


  if (
    profile.quoteCount >= 4
  ) {

    scores.books += 3;
  }


  if (
    profile.dialogueLines >= 2
  ) {

    scores.books += 5;
  }


  if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    scores.books += 2;
  }


  const [
    domain,
    score
  ] =
    Object.entries(
      scores
    )
      .sort(
        (a, b) =>
          b[1] - a[1]
      )[0];


  if (
    !score ||
    score < 4
  ) {

    return {

      domain:
        'general',

      confidence:
        'low',

      score:
        score || 0
    };
  }


  return {

    domain,

    confidence:
      score >= 12
        ? 'high'
        : score >= 7
          ? 'medium'
          : 'low',

    score
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
    0.65
  ) {

    score += 22;

    reasons.push(
      'high sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score += 14;

    reasons.push(
      'moderate sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.30
  ) {

    score += 6;
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

    score += 14;

    reasons.push(
      'dialogue / quotation structure'
    );

  } else if (
    profile.quoteCount >= 3
  ) {

    score += 6;
  }


  if (
    profile.firstPerson >= 4
  ) {

    score += 8;

    reasons.push(
      'personal voice'
    );

  } else if (
    profile.firstPerson > 0
  ) {

    score += 4;
  }


  if (
    profile.contractions >= 4
  ) {

    score += 5;
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
    profile.lexicalDiversity >= 0.62
  ) {

    score += 6;
  }


  if (
    profile.titleReferences >= 2
  ) {

    score += 8;

    reasons.push(
      'literary character references'
    );
  }


  if (
    profile.literaryVocabulary >= 3
  ) {

    score += 10;

    reasons.push(
      'literary prose characteristics'
    );
  }


  if (
    profile.semicolonCount >= 3 &&
    profile.averageSentenceLength >= 24
  ) {

    score += 8;

    reasons.push(
      'complex literary sentence structure'
    );
  }


  if (
    profile.transitions >= 4
  ) {

    score -= 6;
  }


  if (
    domain === 'books'
  ) {

    score += 14;

    reasons.push(
      'literary-domain protection'
    );
  }


  if (
    domain === 'poetry'
  ) {

    score += 12;

    reasons.push(
      'poetry-domain protection'
    );
  }


  return {

    score:
      clamp(
        Math.round(score)
      ),

    reasons:
      [
        ...new Set(reasons)
      ]
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
      ).length > maxChars &&
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
    'Deep verification: loading Model C…'
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

    return (
      1 -
      human
    );
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
    aiProbability(
      result
    ) * 100
  );
}


/* =========================================================
   MODEL C ROUTING
========================================================= */

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
      ? Math.max(...segmentScores) -
        Math.min(...segmentScores)
      : 0;


  const segmentSD =
    standardDeviation(
      segmentScores
    );


  return (

    words < 180 ||

    modelGap >= 18 ||

    (
      raw >= 30 &&
      raw <= 88
    ) ||

    segmentRange >= 40 ||

    segmentSD >= 18 ||

    humanScore >= 40 ||

    domain === 'books' ||

    domain === 'poetry'
  );
}
/* =========================================================
   BENCHMARK STORAGE + MIGRATION
========================================================= */

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


function normalizeLegacyRecord(
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


  const scores =
    record.scores || {

      tmr:
        record.models?.tmr ??
        record.tmr ??
        NaN,

      e5:
        record.models?.e5 ??
        record.e5 ??
        NaN,

      modern:
        record.models?.modern ??
        record.modern ??
        NaN
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
        `AI Trace benchmark migrated from ${legacyKey} to ${BENCH_KEY}`
      );


      return records;
    }
  }


  return [];
}


function saveBench(
  records
) {

  return saveJSON(
    BENCH_KEY,
    records
  );
}


/* =========================================================
   PREDICTION HELPERS
========================================================= */

function verdictToPrediction(
  verdict
) {

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


/* =========================================================
   BENCHMARK METRICS
========================================================= */

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


/* =========================================================
   INDIVIDUAL DETECTOR PERFORMANCE
========================================================= */

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


/* =========================================================
   DOMAIN BENCHMARK
========================================================= */

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


/* =========================================================
   BENCHMARK READINESS
========================================================= */

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


/* =========================================================
   DETECTOR RELIABILITY WEIGHTS
========================================================= */

function detectorReliability(
  detectorName,
  records = loadBench()
) {

  const metrics =
    detectorMetrics(
      detectorName,
      records
    );


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
    metrics.coverage /
    100;


  const accuracy =
    metrics.accuracy /
    100;


  const falsePositivePenalty =
    metrics.fpr /
    100;


  const falseNegativePenalty =
    metrics.fnr /
    100;


  let reliability =

    accuracy *
    0.55 +

    coverage *
    0.20 +

    (
      1 -
      falsePositivePenalty
    ) *
    0.15 +

    (
      1 -
      falseNegativePenalty
    ) *
    0.10;


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


/* =========================================================
   BENCHMARK WEIGHTS
========================================================= */

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


/* =========================================================
   WEIGHTED MEDIAN-LIKE CONSENSUS
========================================================= */

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
  ]
    .filter(
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


/* =========================================================
   CONSENSUS ENGINE V5.3
========================================================= */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  human,
  thirdUsed
}) {

  const active =
    [
      scores.tmr,
      scores.e5,
      scores.modern
    ]
      .filter(
        Number.isFinite
      );


  const weights =
    benchmarkWeights();


  const medianSignal =
    median(
      active
    );


  const weightedSignal =
    weightedAverageScores(
      scores,
      weights
    );


  /*
    Benchmark weighting is intentionally
    conservative.

    Until enough real known samples exist,
    the benchmark does NOT influence scans.
  */

  const raw =
    Math.round(

      weights.active

        ? medianSignal *
            0.55 +

          weightedSignal *
            0.45

        : medianSignal
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
    Math.round(
      standardDeviation(
        segmentScores
      )
    );


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

    domainRisk = 8;

  } else {

    domainRisk = 5;
  }


  let quality = 100;


  /*
    Model disagreement
  */

  quality -=
    Math.min(
      38,
      modelSpread *
      0.75
    );


  /*
    Segment instability
  */

  quality -=
    Math.min(
      30,

      segmentSD *
      0.65 +

      segmentRange *
      0.18
    );


  /*
    Text length
  */

  quality -=
    profile.words < 120

      ? 20

      : profile.words < 180

        ? 13

        : profile.words < 250

          ? 7

          : 0;


  /*
    Language support
  */

  quality -=
    language === 'English'
      ? 0
      : 35;


  /*
    Domain uncertainty
  */

  quality -=
    domainRisk;


  /*
    Two-model mode is allowed,
    but receives a small quality penalty.
  */

  quality -=
    thirdUsed
      ? 0
      : 7;


  /*
    If only one model worked,
    quality must be heavily restricted.
  */

  if (
    active.length < 2
  ) {

    quality -= 35;
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


  /*
    HUMAN COUNTER-EVIDENCE

    This does not independently declare
    a document human.

    It reduces an AI score only when the
    detector evidence itself is unstable.
  */

  const humanPenalty =
    human.score *
    (
      0.08 +
      instability *
      0.48
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


  /* =======================================================
     CONFLICT DEFENSE
  ======================================================= */

  const highModelConflict =
    modelSpread >= 32;


  const severeModelConflict =
    modelSpread >= 48;


  const highSegmentConflict =
    segmentRange >= 65 ||
    segmentSD >= 26;


  const literaryRisk =
    (
      domain === 'books' ||
      domain === 'poetry'
    );


  const humanConflict =
    raw >= 68 &&
    human.score >= 48;


  /*
    Literary false-positive protection.
  */

  if (
    literaryRisk &&
    human.score >= 42 &&
    (
      modelSpread >= 18 ||
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
    Severe detector disagreement must
    never produce a strong accusation.
  */

  if (
    severeModelConflict
  ) {

    calibrated =
      clamp(
        calibrated,
        25,
        75
      );
  }


  const uncertainty =
    clamp(
      Math.round(

        (
          100 -
          quality
        ) +

        (
          humanConflict
            ? (
                human.score -
                40
              ) *
              0.35
            : 0
        ) +

        (
          severeModelConflict
            ? 12
            : 0
        )
      ),

      5,
      95
    );


  const confidence =
    100 -
    uncertainty;


  let verdict =
    'INCONCLUSIVE';


  /*
    Strong AI requires:
    - English
    - multiple working models
    - high calibrated score
    - strong evidence quality
    - no major conflict
    - low human counter-evidence
  */

  if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated >= 87 &&
    quality >= 76 &&
    modelSpread < 24 &&
    segmentRange < 52 &&
    human.score < 45
  ) {

    verdict =
      'Strong AI evidence';
  }


  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated >= 74 &&
    quality >= 61 &&
    modelSpread < 30 &&
    segmentRange < 62 &&
    human.score < 55
  ) {

    verdict =
      'Likely AI';
  }


  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated <= 18 &&
    quality >= 58 &&
    human.score >= 45 &&
    modelSpread < 28
  ) {

    verdict =
      'Strong human evidence';
  }


  else if (
    language === 'English' &&
    active.length >= 2 &&
    calibrated <= 36 &&
    human.score >= 40 &&
    modelSpread < 30
  ) {

    verdict =
      'Likely human';
  }


  /*
    Final abstention guards.
  */

  if (
    language !== 'English' ||
    active.length < 2 ||
    severeModelConflict
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  if (
    verdict.includes('AI') &&
    (
      highModelConflict ||
      highSegmentConflict ||
      humanConflict
    )
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  if (
    verdict.includes('AI') &&
    literaryRisk &&
    human.score >= 42
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  return {

    raw,

    medianSignal:
      Math.round(
        medianSignal
      ),

    weightedSignal:
      Math.round(
        weightedSignal
      ),

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

    modelSpread:
      Math.round(
        modelSpread
      ),

    modelSD,

    segmentRange:
      Math.round(
        segmentRange
      ),

    segmentSD,

    activeModels:
      active.length,

    thirdUsed,

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    highModelConflict,

    severeModelConflict,

    highSegmentConflict,

    humanConflict,

    literaryRisk,

    benchmarkWeightsActive:
      weights.active,

    benchmarkReadiness:
      weights.readiness.level,

    weights: {

      tmr:
        Number(
          weights.tmr.toFixed(3)
        ),

      e5:
        Number(
          weights.e5.toFixed(3)
        ),

      modern:
        Number(
          weights.modern.toFixed(3)
        )
    }
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


    setState(
      'Smart Scan running…'
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


    /* =====================================================
       MODEL A — TMR
    ===================================================== */

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


        const segmentScore =
          await classify(
            modelA,
            chunks[i]
          );


        segmentScores.push(
          segmentScore
        );
      }

    } catch (error) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /* =====================================================
       MODEL B — E5
    ===================================================== */

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

    } catch (error) {

      console.error(
        'E5 failed:',
        error
      );
    }


    const mobile =
      isMobileDevice();


    /*
      ModernBERT is conditional on desktop.

      Mobile remains two-model by design
      because browser memory is more limited.
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


    /* =====================================================
       MODEL C — MODERNBERT
    ===================================================== */

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
      If segment analysis failed,
      do not crash calibration.
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
      90,
      'Calibrating evidence…'
    );


    /*
      Yield to the browser before calibration/render.
      This helps the progress label paint correctly
      instead of appearing frozen.
    */

    await new Promise(
      resolve =>
        requestAnimationFrame(
          () =>
            resolve()
        )
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
      94,
      'Building evidence report…'
    );


    try {

      render(
        scan
      );

    } catch (renderError) {

      console.error(
        'Render failed:',
        renderError
      );


      throw new Error(
        `Analysis completed but report rendering failed: ${renderError.message}`
      );
    }


    try {

      saveHistory(
        scan
      );

    } catch (historyError) {

      console.warn(
        'History save failed:',
        historyError
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
        'V5.3 Mobile Safe • Smart Consensus ✓'
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V5.3 Deep Consensus • 3 models ✓'
      );

    } else {

      setState(
        'V5.3 Smart Consensus • 2 models ✓'
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
            'Benchmark prompt failed:',
            error
          );
        }

      },
      650
    );

  } catch (fatalError) {

    console.error(
      'AI TRACE V5.3 fatal error:',
      fatalError
    );


    setState(
      'Scan error'
    );


    alert(
      `AI Trace could not complete the scan.

${fatalError?.message || 'Unknown error'}

Open the browser console for technical details.`
    );

  } finally {

    $('scan').disabled =
      false;


    hideProgress();
  }
}
/* =========================================================
   RENDER REPORT
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

    const benchmarkText =
      consensus.benchmarkWeightsActive
        ? ` Benchmark calibration is active (${consensus.benchmarkReadiness}).`
        : ' Benchmark calibration is not yet active because more known samples are required.';


    if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {

      $('explain').textContent =
        `AI Trace abstained because the evidence was not reliable enough for a confident AI/Human classification. Diagnostic AI signal: ${consensus.calibrated}%. Evidence quality: ${consensus.quality}%.${benchmarkText}`;

    } else {

      $('explain').textContent =
        `Raw detector consensus ${consensus.raw}%; calibrated to ${consensus.calibrated}% after model agreement, segment stability, domain context and human counter-evidence.${benchmarkText}`;
    }
  }


  /* =======================================================
     TRACE DNA
  ======================================================= */

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

    $('engineBadge').textContent =
      consensus.benchmarkWeightsActive
        ? 'V5.3 • BENCHMARK CALIBRATED'
        : consensus.thirdUsed
          ? 'V5.3 • 3-MODEL CONSENSUS'
          : isMobileDevice()
            ? 'V5.3 • MOBILE SAFE'
            : 'V5.3 • SMART CONSENSUS';
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
        ? 'Disabled on mobile for stability'
        : 'Not required by Smart Scan';


  const weightText =
    consensus.benchmarkWeightsActive
      ? `TMR ${consensus.weights.tmr} · E5 ${consensus.weights.e5} · Modern ${consensus.weights.modern}`
      : 'Waiting for sufficient known benchmark samples';


  const evidence = [

    [
      'Final decision',
      consensus.verdict,
      'Outcome'
    ],

    [
      'Calibrated AI signal',
      `${consensus.calibrated}%`,
      'Primary'
    ],

    [
      'Raw detector consensus',
      `${consensus.raw}%`,
      'Diagnostic'
    ],

    [
      'Median detector signal',
      `${consensus.medianSignal}%`,
      'Diagnostic'
    ],

    [
      'Benchmark-weighted signal',
      `${consensus.weightedSignal}%`,
      consensus.benchmarkWeightsActive
        ? 'Active'
        : 'Inactive'
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
      'Benchmark calibration',
      weightText,
      consensus.benchmarkWeightsActive
        ? consensus.benchmarkReadiness
        : 'Learning'
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
      'Model spread',
      `${consensus.modelSpread} points`,
      consensus.severeModelConflict
        ? 'Severe conflict'
        : consensus.highModelConflict
          ? 'High conflict'
          : 'Acceptable'
    ],

    [
      'Segment range',
      `${consensus.segmentRange} points`,
      consensus.highSegmentConflict
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      consensus.literaryRisk
        ? 'Protected domain'
        : 'Routing'
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

    <span>
      ${escapeHTML(item[0])}
    </span>

    <span>
      ${escapeHTML(item[2])}
    </span>

  </div>

  <small>
    ${escapeHTML(item[1])}
  </small>

</div>

`
        )
        .join('');
  }


  /* =======================================================
     METRICS
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

    'Model spread':
      `${consensus.modelSpread} pts`,

    'Model deviation':
      consensus.modelSD,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Evidence quality':
      `${consensus.quality}%`,

    'Benchmark calibration':
      consensus.benchmarkWeightsActive
        ? consensus.benchmarkReadiness
        : 'Pending',

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


  /* =======================================================
     TRACE MAP
  ======================================================= */

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


/* =========================================================
   BENCHMARK PROMPT
========================================================= */

function benchmarkPrompt(scan) {

  const answer =
    prompt(
`AI TRACE V5.3 BENCHMARK

Only label samples whose true origin you KNOW.

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

These are development measurements only.`
  );
}


/* =========================================================
   BENCHMARK PANEL
========================================================= */

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


  const domainStats =
    domainMetrics(
      records
    );


  const weights =
    benchmarkWeights();


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
    · Weight ${Number(weight).toFixed(2)}
  </small>

</div>

`
    )
    .join('');


  const domainHTML =
    domainStats
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
        record => {

          const score =
            record.consensus
              ?.calibrated ??
            '?';


          const verdict =
            record.consensus
              ?.verdict ??
            'legacy';


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
    ${escapeHTML(
      record.domain ||
      'general'
    )}
    · ${escapeHTML(
      String(score)
    )}% calibrated
    · ${escapeHTML(
      String(
        record.consensus
          ?.quality ??
        '?'
      )
    )}% quality
    · ${escapeHTML(
      verdict
    )}
  </small>

</div>

`;

        }
      )
      .join('');


  panel.innerHTML = `

<span class="over">
  V5.3 BENCHMARK • DEVELOPMENT ONLY
</span>

<h2>
  Benchmark Intelligence
</h2>

<p class="sub">
  Known samples are used to measure each detector independently. Benchmark weights remain disabled until enough AI and HUMAN samples exist.
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
            ? 'WEIGHTING ACTIVE'
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
        No V5.3 benchmark records yet.
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


/* =========================================================
   DEV API
========================================================= */

window.AITraceV53 = {


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
      `AI-Trace-V53-Benchmark-${Date.now()}.json`;


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
        'Delete all V5.3 benchmark records from this device?'
      );


    if (!confirmation) {

      return;
    }


    localStorage.removeItem(
      BENCH_KEY
    );


    renderBenchmarkPanel();


    alert(
      'V5.3 benchmark data deleted.'
    );
  },


  clearHistory() {

    const confirmation =
      confirm(
        'Delete local V5.3 scan history?'
      );


    if (!confirmation) {

      return;
    }


    localStorage.removeItem(
      HISTORY_KEY
    );


    alert(
      'V5.3 scan history deleted.'
    );
  }
};


/* =========================================================
   INIT
========================================================= */

updateCount();


setTimeout(
  renderBenchmarkPanel,
  350
);


console.info(
  `AI TRACE V${VERSION} loaded`
);
