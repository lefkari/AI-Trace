/*
  ============================================================
  AI TRACE V6.2 — ADAPTIVE RELIABILITY ENGINE
  PART 1 / 3

  Core:
  - TMR
  - E5-small
  - Conditional ModernBERT
  - Human counter-evidence
  - Domain detection
  - Segment analysis
  - Benchmark-driven reliability
  - Domain-aware detector reliability
  - Minimum-sample protection
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
   VERSION
============================================================ */

const VERSION = '6.2';


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
   STORAGE
============================================================ */

const BENCH_KEY =
  'aiTraceBenchmarkV62';

const HISTORY_KEY =
  'aiTraceHistoryV62';


const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV61',
  'aiTraceBenchmarkV6',
  'aiTraceBenchmarkV54',
  'aiTraceBenchmarkV53',
  'aiTraceBenchmarkV52',
  'aiTraceBenchmarkV51',
  'aiTraceBenchmarkV44'
];


/* ============================================================
   MODEL CACHE
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
      Number(values[i]);

    const weight =
      Number(weights[i]);

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
    usable[
      middle - 1
    ] +
    usable[
      middle
    ]
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

  return String(value).replace(
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


function nowISO() {

  return new Date()
    .toISOString();
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
    latin /
    total >
    0.82
  )
    ? 'English'
    : 'Non-English';
}


/* ============================================================
   STORAGE HELPERS
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

    if (!raw) {
      return fallback;
    }

    return JSON.parse(
      raw
    );

  } catch (error) {

    console.warn(
      `Could not read ${key}:`,
      error
    );

    return fallback;
  }
}


function saveJSON(
  key,
  data
) {

  try {

    localStorage.setItem(
      key,
      JSON.stringify(
        data
      )
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
   UI
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
      `${clamp(
        percent
      )}%`;
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

  if (
    $('modelState')
  ) {

    $('modelState').textContent =
      label;
  }
}


function loadDemo() {

  if (!textEl) {
    return;
  }

  textEl.value =
`Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

One major advantage of artificial intelligence is its ability to process large amounts of information quickly. Businesses can automate repetitive workflows, researchers can examine large datasets, and individuals can use intelligent tools to improve productivity. These systems can identify patterns that might be difficult for humans to notice manually.

At the same time, artificial intelligence introduces important challenges. Generated content may contain incorrect information, fabricated details, or biased conclusions. As AI-generated text becomes more natural, it can also become increasingly difficult to determine whether a document was written by a person or produced by a machine.

Reliable AI detection therefore requires more than a single score. A responsible system should combine independent signals, communicate uncertainty, evaluate disagreement, test itself on known samples, and avoid making strong claims when the available evidence is weak.`;

  updateCount();
}


/* ============================================================
   UI EVENTS
============================================================ */

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
      .add(
        'hidden'
      );
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
    punctuationPatterns.filter(
      regex =>
        (
          value.match(
            regex
          ) || []
        ).length > 0
    ).length;


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


  const subjectiveMarkers =
    countMatches(
      value,
      /\b(I think|I believe|I suppose|I feel|in my view|perhaps|maybe|it seems to me|I do not know|I remember)\b/gi
    );


  const transitions =
    countMatches(
      value,
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result|on the other hand)\b/gi
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

    semicolonCount,

    dialogueLines,

    firstPerson,

    contractions,

    subjectiveMarkers,

    transitions,

    titleReferences,

    narrativeMarkers,

    literaryVocabulary
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


  const scores = {

    academic:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract|research|findings|methodology)\b/g
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
        /\b(aita|tldr|subreddit|upvote|downvote|throwaway|imo|lol|op)\b/g
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
    profile.sentenceBurstiness >= 0.45
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
        score || 0,

      scores
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

    score,

    scores
  };
}


/* ============================================================
   HUMAN COUNTER-EVIDENCE
============================================================ */

function calculateHumanEvidence(
  profile,
  domain
) {

  let score = 0;

  const reasons = [];


  if (
    profile.sentenceBurstiness >= 0.68
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
    profile.quoteCount >= 8 ||
    profile.dialogueLines >= 2
  ) {

    score += 16;

    reasons.push(
      'Dialogue or quotation structure'
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
      'Personal or subjective voice'
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
      'Natural contraction usage'
    );
  }


  if (
    profile.paragraphDeviation >= 25 &&
    profile.paragraphs >= 3
  ) {

    score += 8;

    reasons.push(
      'Irregular paragraph rhythm'
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
    profile.semicolonCount >= 2
  ) {

    score += 5;

    reasons.push(
      'Complex punctuation rhythm'
    );
  }


  if (
    profile.titleReferences >= 2
  ) {

    score += 7;

    reasons.push(
      'Character-reference structure'
    );
  }


  if (
    profile.literaryVocabulary >= 3
  ) {

    score += 8;

    reasons.push(
      'Literary vocabulary'
    );
  }


  if (
    profile.transitions >= 5
  ) {

    score -= 7;
  }


  if (
    domain === 'books'
  ) {

    score += 12;

    reasons.push(
      'Literary-domain protection'
    );
  }


  if (
    domain === 'poetry'
  ) {

    score += 14;

    reasons.push(
      'Poetry-domain protection'
    );
  }


  return {

    score:
      clamp(
        Math.round(
          score
        )
      ),

    reasons:
      [
        ...new Set(
          reasons
        )
      ]
  };
}


/* ============================================================
   TEXT CHUNKING
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
    'Loading Model C…'
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
        item?.label || ''
      )
        .toLowerCase()
        .trim();


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

    return clamp(
      ai,
      0,
      1
    );
  }


  if (
    human !== null
  ) {

    return clamp(
      1 - human,
      0,
      1
    );
  }


  if (
    results.length >= 2
  ) {

    const second =
      Number(
        results[1]?.score
      );


    if (
      Number.isFinite(
        second
      )
    ) {

      return clamp(
        second,
        0,
        1
      );
    }
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
    ) *
    100
  );
}


/* ============================================================
   BENCHMARK RECORD NORMALIZATION
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

    consensus:
      record.consensus || {

        calibrated:
          record.calibratedScore ??
          record.score ??
          record.rawSignal ??
          50,

        quality:
          record.evidenceQuality ??
          record.confidence ??
          50,

        verdict:
          record.verdict ||
          'INCONCLUSIVE'
      }
  };
}


/* ============================================================
   BENCHMARK LOAD + MIGRATION
============================================================ */

function loadBench() {

  const current =
    loadJSON(
      BENCH_KEY,
      []
    );


  if (
    Array.isArray(
      current
    ) &&
    current.length
  ) {

    return current
      .map(
        normalizeBenchmarkRecord
      )
      .filter(Boolean);
  }


  for (
    const legacyKey
    of LEGACY_BENCH_KEYS
  ) {

    const legacy =
      loadJSON(
        legacyKey,
        []
      );


    if (
      Array.isArray(
        legacy
      ) &&
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


      console.info(
        `AI Trace V6.2 migrated benchmark from ${legacyKey}`
      );


      return normalized;
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


/* ============================================================
   BENCHMARK PREDICTION
============================================================ */

function benchmarkPrediction(
  record
) {

  const verdict =
    record?.consensus
      ?.verdict ||
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


/* ============================================================
   DETECTOR RAW PREDICTION
============================================================ */

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
   GENERIC CONFUSION METRICS
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

function detectorMetrics(
  detectorName,
  records = loadBench()
) {

  const rows =
    binaryRecords(
      records
    )
      .filter(
        record =>
          Number.isFinite(
            Number(
              record.scores?.[
                detectorName
              ]
            )
          )
      );


  return evaluatePredictions(
    rows,

    record =>
      detectorPrediction(
        Number(
          record.scores?.[
            detectorName
          ]
        )
      )
  );
}


/* ============================================================
   GLOBAL DETECTOR RELIABILITY
============================================================ */

function globalDetectorReliability(
  detectorName,
  records = loadBench()
) {

  const metrics =
    detectorMetrics(
      detectorName,
      records
    );


  /*
    MINIMUM SAMPLE GUARD

    With tiny datasets we do NOT allow
    benchmark measurements to dominate.
  */

  if (
    metrics.total < 20
  ) {

    return {

      ready:
        false,

      samples:
        metrics.total,

      weight:
        1,

      metrics
    };
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


  let reliability =
    accuracy * 0.45 +
    coverage * 0.15 +
    fprSafety * 0.27 +
    fnrSafety * 0.13;


  /*
    False positives are especially dangerous
    for an AI-authorship detector.
  */

  if (
    metrics.fpr >= 40
  ) {

    reliability *= 0.70;
  }


  if (
    metrics.fpr >= 70
  ) {

    reliability *= 0.55;
  }


  reliability =
    clamp(
      reliability,
      0.20,
      1
    );


  return {

    ready:
      true,

    samples:
      metrics.total,

    weight:
      Number(
        reliability.toFixed(
          3
        )
      ),

    metrics
  };
}


/* ============================================================
   DOMAIN-SPECIFIC DETECTOR RELIABILITY
============================================================ */

function domainDetectorReliability(
  detectorName,
  domain,
  records = loadBench()
) {

  const domainRows =
    binaryRecords(
      records
    )
      .filter(
        record =>
          (
            record.domain ||
            'general'
          ) === domain
      );


  const metrics =
    detectorMetrics(
      detectorName,
      domainRows
    );


  /*
    Domain reliability requires even more caution.
    We do not activate it until there are enough
    samples in that specific domain.
  */

  if (
    metrics.total < 12 ||
    metrics.totalAI < 4 ||
    metrics.totalHuman < 4
  ) {

    return {

      ready:
        false,

      samples:
        metrics.total,

      weight:
        1,

      metrics
    };
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


  let reliability =
    accuracy * 0.40 +
    coverage * 0.12 +
    fprSafety * 0.33 +
    fnrSafety * 0.15;


  if (
    metrics.fpr >= 40
  ) {

    reliability *= 0.65;
  }


  if (
    metrics.fpr >= 70
  ) {

    reliability *= 0.50;
  }


  reliability =
    clamp(
      reliability,
      0.15,
      1
    );


  return {

    ready:
      true,

    samples:
      metrics.total,

    weight:
      Number(
        reliability.toFixed(
          3
        )
      ),

    metrics
  };
}


/* ============================================================
   ADAPTIVE RELIABILITY PROFILE
============================================================ */

function adaptiveReliabilityProfile(
  domain,
  records = loadBench()
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

    const global =
      globalDetectorReliability(
        detector,
        records
      );


    const domainSpecific =
      domainDetectorReliability(
        detector,
        domain,
        records
      );


    let weight = 1;


    if (
      global.ready &&
      domainSpecific.ready
    ) {

      /*
        Domain reliability receives slightly
        more influence than global reliability.
      */

      weight =
        global.weight * 0.42 +
        domainSpecific.weight * 0.58;

    } else if (
      global.ready
    ) {

      weight =
        global.weight;

    } else {

      /*
        Tiny benchmark:
        do not pretend we have learned reliability.
      */

      weight = 1;
    }


    result[
      detector
    ] = {

      weight:
        Number(
          clamp(
            weight,
            0.15,
            1.15
          ).toFixed(
            3
          )
        ),

      global,

      domain:
        domainSpecific
    };
  }


  return result;
}
/* ============================================================
   AI TRACE V6.2
   PART 2 / 3

   - Directional detector reliability
   - Adaptive model weighting
   - Outlier defense
   - Segment reliability
   - Evidence sufficiency
   - Third-model routing
   - Adaptive consensus
   - Calibration
   - Abstention protection
============================================================ */


/* ============================================================
   DIRECTIONAL DETECTOR RELIABILITY
============================================================ */

function directionalDetectorReliability(
  detectorName,
  direction,
  domain,
  records = loadBench()
) {

  const binary =
    binaryRecords(
      records
    );


  const globalRows =
    binary.filter(
      record =>
        Number.isFinite(
          Number(
            record.scores?.[
              detectorName
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


  function calculate(rows) {

    let relevant = 0;
    let correct = 0;
    let wrong = 0;


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


      const prediction =
        detectorPrediction(
          score
        );


      if (
        prediction !==
        direction
      ) {

        continue;
      }


      relevant++;


      if (
        prediction ===
        record.truth
      ) {

        correct++;

      } else {

        wrong++;
      }
    }


    return {

      samples:
        relevant,

      correct,

      wrong,

      accuracy:
        percentage(
          correct,
          relevant
        )
    };
  }


  const global =
    calculate(
      globalRows
    );


  const domainSpecific =
    calculate(
      domainRows
    );


  /*
    We require enough directional examples before
    allowing benchmark learning to influence a model.

    Example:
    A detector should not receive a huge HUMAN weight
    because it happened to classify two human samples
    correctly.
  */

  const globalReady =
    global.samples >= 10;


  const domainReady =
    domainSpecific.samples >= 6;


  let reliability = 1;


  if (
    globalReady
  ) {

    reliability =
      clamp(
        global.accuracy /
        100,
        0.20,
        1
      );
  }


  if (
    globalReady &&
    domainReady
  ) {

    reliability =
      reliability * 0.45 +
      clamp(
        domainSpecific.accuracy /
        100,
        0.15,
        1
      ) * 0.55;

  } else if (
    !globalReady &&
    domainReady
  ) {

    /*
      Domain-only learning is intentionally
      conservative.
    */

    reliability =
      0.65 +
      (
        clamp(
          domainSpecific.accuracy /
          100,
          0.15,
          1
        ) -
        0.65
      ) * 0.35;
  }


  return {

    ready:
      globalReady ||
      domainReady,

    weight:
      Number(
        clamp(
          reliability,
          0.15,
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
   FULL MODEL RELIABILITY
============================================================ */

function buildModelReliability(
  domain,
  records = loadBench()
) {

  const base =
    adaptiveReliabilityProfile(
      domain,
      records
    );


  const result = {};


  for (
    const detector
    of [
      'tmr',
      'e5',
      'modern'
    ]
  ) {

    result[
      detector
    ] = {

      base:
        base[
          detector
        ]?.weight ??
        1,

      ai:
        directionalDetectorReliability(
          detector,
          'AI',
          domain,
          records
        ),

      human:
        directionalDetectorReliability(
          detector,
          'HUMAN',
          domain,
          records
        ),

      global:
        base[
          detector
        ]?.global,

      domain:
        base[
          detector
        ]?.domain
    };
  }


  return result;
}


/* ============================================================
   MODEL WEIGHT FOR CURRENT SCORE
============================================================ */

function modelWeightForScore(
  detectorName,
  score,
  reliability
) {

  if (
    !Number.isFinite(
      score
    )
  ) {

    return 0;
  }


  const detector =
    reliability?.[
      detectorName
    ];


  if (!detector) {
    return 1;
  }


  const base =
    detector.base ??
    1;


  let directional = 1;


  if (
    score >= 70
  ) {

    directional =
      detector.ai?.weight ??
      1;

  } else if (
    score <= 30
  ) {

    directional =
      detector.human?.weight ??
      1;

  } else {

    /*
      Scores near the middle are uncertain.
      Directional benchmark performance should
      therefore have little influence.
    */

    const ai =
      detector.ai?.weight ??
      1;


    const human =
      detector.human?.weight ??
      1;


    directional =
      (
        ai +
        human
      ) / 2;
  }


  /*
    Directional reliability receives less influence
    than the global/domain reliability because
    directional datasets are normally smaller.
  */

  const combined =
    base * 0.68 +
    directional * 0.32;


  return Number(
    clamp(
      combined,
      0.15,
      1.15
    ).toFixed(
      3
    )
  );
}


/* ============================================================
   OUTLIER ANALYSIS
============================================================ */

function analyzeModelOutliers(
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
        0,

      reference:
        median(
          entries.map(
            entry =>
              entry[1]
          )
        )
    };
  }


  const values =
    entries.map(
      entry =>
        entry[1]
    );


  const reference =
    median(
      values
    );


  const distances =
    entries.map(
      (
        [
          name,
          value
        ]
      ) => ({

        name,

        value,

        distance:
          Math.abs(
            value -
            reference
          )
      })
    );


  distances.sort(
    (
      a,
      b
    ) =>
      b.distance -
      a.distance
  );


  const largest =
    distances[0];


  const second =
    distances[1];


  /*
    We only call something an outlier when:

    1. It is far away from consensus.
    2. It is substantially farther than the
       next-most-distant detector.
  */

  const detected =
    largest.distance >= 30 &&
    largest.distance -
      second.distance >= 15;


  return {

    detected,

    detector:
      detected
        ? largest.name
        : null,

    distance:
      Math.round(
        largest.distance
      ),

    reference:
      Math.round(
        reference
      )
  };
}


/* ============================================================
   ADAPTIVE WEIGHTED MODEL SIGNAL
============================================================ */

function adaptiveModelSignal(
  scores,
  reliability
) {

  const outlier =
    analyzeModelOutliers(
      scores
    );


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
      modelWeightForScore(
        detector,
        score,
        reliability
      );


    /*
      A detector identified as a strong isolated
      outlier is not deleted. Its influence is
      reduced instead.
    */

    if (
      outlier.detected &&
      outlier.detector ===
        detector
    ) {

      weight *= 0.38;
    }


    /*
      Extreme predictions receive a small bonus
      only when they agree with the broader ensemble.
    */

    const ensembleReference =
      outlier.reference;


    const agrees =
      Math.abs(
        score -
        ensembleReference
      ) <= 18;


    if (
      agrees &&
      (
        score >= 90 ||
        score <= 10
      )
    ) {

      weight *= 1.05;
    }


    weight =
      clamp(
        weight,
        0.10,
        1.20
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
        outlier.detector ===
          detector
    };
  }


  const signal =
    Math.round(
      weightedAverage(
        values,
        weights
      )
    );


  return {

    signal,

    details,

    outlier
  };
}


/* ============================================================
   SEGMENT ANALYSIS
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

      count:
        0,

      mean:
        50,

      median:
        50,

      min:
        50,

      max:
        50,

      range:
        0,

      deviation:
        0,

      aiSegments:
        0,

      humanSegments:
        0,

      uncertainSegments:
        0,

      mixed:
        false,

      stability:
        0
    };
  }


  const minimum =
    Math.min(
      ...usable
    );


  const maximum =
    Math.max(
      ...usable
    );


  const range =
    maximum -
    minimum;


  const deviation =
    standardDeviation(
      usable
    );


  const aiSegments =
    usable.filter(
      score =>
        score >= 70
    ).length;


  const humanSegments =
    usable.filter(
      score =>
        score <= 30
    ).length;


  const uncertainSegments =
    usable.length -
    aiSegments -
    humanSegments;


  const mixed =
    aiSegments > 0 &&
    humanSegments > 0;


  /*
    Segment stability:
    100 = very consistent document
    0   = extremely unstable document
  */

  let stability =
    100;


  stability -=
    Math.min(
      45,
      deviation * 1.35
    );


  stability -=
    Math.min(
      35,
      range * 0.35
    );


  if (
    mixed
  ) {

    stability -= 12;
  }


  stability =
    clamp(
      Math.round(
        stability
      )
    );


  return {

    count:
      usable.length,

    mean:
      Math.round(
        average(
          usable
        )
      ),

    median:
      Math.round(
        median(
          usable
        )
      ),

    min:
      Math.round(
        minimum
      ),

    max:
      Math.round(
        maximum
      ),

    range:
      Math.round(
        range
      ),

    deviation:
      Math.round(
        deviation
      ),

    aiSegments,

    humanSegments,

    uncertainSegments,

    mixed,

    stability
  };
}


/* ============================================================
   EVIDENCE SUFFICIENCY
============================================================ */

function calculateEvidenceSufficiency({
  words,
  language,
  domain,
  domainConfidence,
  activeModels,
  modelSpread,
  segmentAnalysis,
  humanScore,
  thirdUsed
}) {

  let score = 100;

  const penalties = [];


  /*
    DOCUMENT LENGTH
  */

  if (
    words < 100
  ) {

    score -= 32;

    penalties.push(
      'Very short document'
    );

  } else if (
    words < 150
  ) {

    score -= 20;

    penalties.push(
      'Short document'
    );

  } else if (
    words < 250
  ) {

    score -= 8;
  }


  /*
    LANGUAGE SUPPORT
  */

  if (
    language !==
    'English'
  ) {

    score -= 38;

    penalties.push(
      'Language outside strongest supported path'
    );
  }


  /*
    MODEL AVAILABILITY
  */

  if (
    activeModels <= 1
  ) {

    score -= 38;

    penalties.push(
      'Insufficient model redundancy'
    );

  } else if (
    activeModels === 2
  ) {

    score -= 8;
  }


  /*
    MODEL DISAGREEMENT
  */

  if (
    modelSpread >= 45
  ) {

    score -= 30;

    penalties.push(
      'Severe model disagreement'
    );

  } else if (
    modelSpread >= 30
  ) {

    score -= 18;

    penalties.push(
      'High model disagreement'
    );

  } else if (
    modelSpread >= 20
  ) {

    score -= 8;
  }


  /*
    SEGMENT INSTABILITY
  */

  if (
    segmentAnalysis.range >= 70
  ) {

    score -= 24;

    penalties.push(
      'Severe segment variation'
    );

  } else if (
    segmentAnalysis.range >= 50
  ) {

    score -= 14;

    penalties.push(
      'High segment variation'
    );
  }


  if (
    segmentAnalysis.deviation >= 30
  ) {

    score -= 16;

  } else if (
    segmentAnalysis.deviation >= 20
  ) {

    score -= 8;
  }


  if (
    segmentAnalysis.mixed
  ) {

    score -= 12;

    penalties.push(
      'Mixed AI/Human segment pattern'
    );
  }


  /*
    DOMAIN RISK
  */

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {

    score -= 16;

    penalties.push(
      'High-risk literary domain'
    );

  } else if (
    domainConfidence === 'low'
  ) {

    score -= 5;
  }


  /*
    CONFLICT BETWEEN HUMAN STYLE AND AI SIGNAL
  */

  if (
    humanScore >= 60
  ) {

    score -= 8;
  }


  /*
    Third model can improve sufficiency when it
    was specifically triggered to resolve ambiguity.
  */

  if (
    thirdUsed &&
    activeModels >= 3
  ) {

    score += 6;
  }


  score =
    clamp(
      Math.round(
        score
      )
    );


  return {

    score,

    level:
      score >= 75
        ? 'Strong'
        : score >= 55
          ? 'Moderate'
          : score >= 35
            ? 'Weak'
            : 'Insufficient',

    penalties
  };
}


/* ============================================================
   THIRD MODEL ROUTING
============================================================ */

function shouldUseThirdModel({
  scores,
  segmentScores,
  domain,
  humanScore,
  words,
  language
}) {

  if (
    isMobileDevice()
  ) {

    return false;
  }


  if (
    language !==
    'English'
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
    median(
      [
        scores.tmr,
        scores.e5
      ]
    );


  const segment =
    analyzeSegments(
      segmentScores
    );


  /*
    ModernBERT is not loaded for every scan.

    It is used when the first two models leave
    meaningful ambiguity.
  */

  return (

    words < 180 ||

    gap >= 18 ||

    (
      quickMedian >= 30 &&
      quickMedian <= 80
    ) ||

    segment.range >= 45 ||

    segment.deviation >= 20 ||

    segment.mixed ||

    humanScore >= 45 ||

    domain === 'books' ||

    domain === 'poetry'
  );
}


/* ============================================================
   MODEL AGREEMENT
============================================================ */

function calculateModelAgreement(
  scores
) {

  const active =
    Object.values(
      scores
    )
      .filter(
        Number.isFinite
      );


  if (
    active.length <= 1
  ) {

    return {

      active:
        active.length,

      spread:
        100,

      deviation:
        50,

      agreement:
        0
    };
  }


  const spread =
    Math.max(
      ...active
    ) -
    Math.min(
      ...active
    );


  const deviation =
    standardDeviation(
      active
    );


  let agreement =
    100;


  agreement -=
    Math.min(
      60,
      spread * 1.15
    );


  agreement -=
    Math.min(
      30,
      deviation * 0.65
    );


  agreement =
    clamp(
      Math.round(
        agreement
      )
    );


  return {

    active:
      active.length,

    spread:
      Math.round(
        spread
      ),

    deviation:
      Math.round(
        deviation
      ),

    agreement
  };
}


/* ============================================================
   DOMAIN-SPECIFIC CALIBRATION
============================================================ */

function domainCalibration({
  signal,
  domain,
  humanScore,
  modelAgreement,
  segmentAnalysis
}) {

  let calibrated =
    signal;


  /*
    Literary material is one of the most dangerous
    domains for false positives.

    We do not simply declare literary text human.
    Instead, strong AI signals require stronger
    supporting evidence.
  */

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {

    if (
      humanScore >= 55
    ) {

      calibrated -=
        Math.min(
          18,
          (
            humanScore -
            45
          ) * 0.35
        );
    }


    if (
      modelAgreement.spread >= 25
    ) {

      calibrated -= 7;
    }


    if (
      segmentAnalysis.range >= 45
    ) {

      calibrated -= 6;
    }
  }


  /*
    Highly structured informational writing can
    naturally resemble generated text, so extreme
    confidence is softened slightly.
  */

  if (
    domain === 'academic' ||
    domain === 'encyclopedia' ||
    domain === 'news'
  ) {

    if (
      calibrated >= 80 &&
      humanScore >= 35
    ) {

      calibrated -= 4;
    }
  }


  return clamp(
    Math.round(
      calibrated
    )
  );
}


/* ============================================================
   HUMAN COUNTER-EVIDENCE CALIBRATION
============================================================ */

function applyHumanCounterEvidence({
  signal,
  humanScore,
  evidenceQuality,
  domain
}) {

  /*
    Human-style signals must never directly flip a
    strong detector result.

    Their influence increases when model evidence
    itself is uncertain.
  */

  const uncertaintyFactor =
    1 -
    evidenceQuality /
    100;


  let strength =
    0.08 +
    uncertaintyFactor * 0.32;


  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {

    strength += 0.08;
  }


  const penalty =
    humanScore *
    strength *
    (
      signal /
      100
    );


  return {

    penalty:
      Math.round(
        penalty
      ),

    score:
      clamp(
        Math.round(
          signal -
          penalty
        )
      )
  };
}


/* ============================================================
   EXTREME SCORE PROTECTION
============================================================ */

function protectExtremeScore({
  score,
  modelAgreement,
  segmentAnalysis,
  sufficiency,
  humanScore
}) {

  let result =
    score;


  /*
    A score above 95 should only survive when the
    evidence is exceptionally consistent.
  */

  if (
    result >= 95
  ) {

    if (
      modelAgreement.spread > 15 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 80 ||
      humanScore >= 45
    ) {

      result =
        Math.min(
          result,
          94
        );
    }
  }


  /*
    Same principle on the human side.
  */

  if (
    result <= 5
  ) {

    if (
      modelAgreement.spread > 15 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 75
    ) {

      result =
        Math.max(
          result,
          6
        );
    }
  }


  return clamp(
    Math.round(
      result
    )
  );
}


/* ============================================================
   VERDICT ENGINE
============================================================ */

function determineVerdict({
  score,
  sufficiency,
  modelAgreement,
  segmentAnalysis,
  humanScore,
  language,
  domain
}) {

  /*
    LANGUAGE ABSTENTION
  */

  if (
    language !==
    'English'
  ) {

    return 'INCONCLUSIVE';
  }


  /*
    INSUFFICIENT EVIDENCE
  */

  if (
    sufficiency.score < 40
  ) {

    return 'INCONCLUSIVE';
  }


  /*
    SEVERE CONFLICT
  */

  if (
    modelAgreement.spread >= 42 ||
    segmentAnalysis.range >= 75
  ) {

    return 'INCONCLUSIVE';
  }


  /*
    MIXED DOCUMENT
  */

  if (
    segmentAnalysis.mixed &&
    segmentAnalysis.range >= 55
  ) {

    return 'INCONCLUSIVE';
  }


  /*
    STRONG AI
  */

  if (
    score >= 88 &&
    sufficiency.score >= 76 &&
    modelAgreement.spread <= 20 &&
    segmentAnalysis.range <= 45 &&
    humanScore < 48
  ) {

    return 'Strong AI evidence';
  }


  /*
    LIKELY AI
  */

  if (
    score >= 72 &&
    sufficiency.score >= 58 &&
    modelAgreement.spread <= 30 &&
    segmentAnalysis.range <= 60 &&
    humanScore < 58
  ) {

    /*
      Literary domains receive additional protection.
    */

    if (
      (
        domain === 'books' ||
        domain === 'poetry'
      ) &&
      (
        humanScore >= 45 ||
        sufficiency.score < 70
      )
    ) {

      return 'INCONCLUSIVE';
    }


    return 'Likely AI';
  }


  /*
    STRONG HUMAN
  */

  if (
    score <= 18 &&
    sufficiency.score >= 60 &&
    modelAgreement.spread <= 28 &&
    humanScore >= 45
  ) {

    return 'Strong human evidence';
  }


  /*
    LIKELY HUMAN
  */

  if (
    score <= 35 &&
    sufficiency.score >= 48 &&
    modelAgreement.spread <= 32 &&
    humanScore >= 35
  ) {

    return 'Likely human';
  }


  return 'INCONCLUSIVE';
}


/* ============================================================
   CONFIDENCE ENGINE
============================================================ */

function calculateConfidence({
  sufficiency,
  modelAgreement,
  segmentAnalysis,
  verdict,
  activeModels
}) {

  let confidence =
    sufficiency.score * 0.45 +
    modelAgreement.agreement * 0.35 +
    segmentAnalysis.stability * 0.20;


  if (
    activeModels <= 1
  ) {

    confidence -= 25;

  } else if (
    activeModels === 2
  ) {

    confidence -= 4;
  }


  if (
    verdict ===
    'INCONCLUSIVE'
  ) {

    /*
      INCONCLUSIVE is a valid result.

      But the UI confidence here describes confidence
      in an AI/Human attribution, not confidence that
      abstention was the right decision.
    */

    confidence =
      Math.min(
        confidence,
        55
      );
  }


  return clamp(
    Math.round(
      confidence
    )
  );
}


/* ============================================================
   MAIN V6.2 CONSENSUS ENGINE
============================================================ */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  domainConfidence,
  human,
  thirdUsed,
  reliability
}) {

  const modelAgreement =
    calculateModelAgreement(
      scores
    );


  const segmentAnalysis =
    analyzeSegments(
      segmentScores
    );


  const adaptive =
    adaptiveModelSignal(
      scores,
      reliability
    );


  /*
    First raw score:
    benchmark-aware weighted detector ensemble.
  */

  const raw =
    adaptive.signal;


  /*
    Evidence sufficiency is calculated before final
    human-style calibration.
  */

  const sufficiency =
    calculateEvidenceSufficiency({

      words:
        profile.words,

      language,

      domain,

      domainConfidence,

      activeModels:
        modelAgreement.active,

      modelSpread:
        modelAgreement.spread,

      segmentAnalysis,

      humanScore:
        human.score,

      thirdUsed
    });


  /*
    Domain calibration
  */

  const domainAdjusted =
    domainCalibration({

      signal:
        raw,

      domain,

      humanScore:
        human.score,

      modelAgreement,

      segmentAnalysis
    });


  /*
    Human counter-evidence
  */

  const humanAdjusted =
    applyHumanCounterEvidence({

      signal:
        domainAdjusted,

      humanScore:
        human.score,

      evidenceQuality:
        sufficiency.score,

      domain
    });


  /*
    Final extreme-score protection
  */

  const calibrated =
    protectExtremeScore({

      score:
        humanAdjusted.score,

      modelAgreement,

      segmentAnalysis,

      sufficiency,

      humanScore:
        human.score
    });


  /*
    Final verdict
  */

  const verdict =
    determineVerdict({

      score:
        calibrated,

      sufficiency,

      modelAgreement,

      segmentAnalysis,

      humanScore:
        human.score,

      language,

      domain
    });


  /*
    Confidence
  */

  const confidence =
    calculateConfidence({

      sufficiency,

      modelAgreement,

      segmentAnalysis,

      verdict,

      activeModels:
        modelAgreement.active
    });


  const uncertainty =
    clamp(
      100 -
      confidence,
      5,
      95
    );


  return {

    raw,

    calibrated,

    verdict,

    confidence,

    uncertainty,

    quality:
      sufficiency.score,

    sufficiency,

    modelAgreement,

    segmentAnalysis,

    reliability,

    modelWeights:
      adaptive.details,

    outlier:
      adaptive.outlier,

    humanPenalty:
      humanAdjusted.penalty,

    activeModels:
      modelAgreement.active,

    thirdUsed,

    modelSpread:
      modelAgreement.spread,

    modelSD:
      modelAgreement.deviation,

    segmentRange:
      segmentAnalysis.range,

    segmentSD:
      segmentAnalysis.deviation
  };
}


/* ============================================================
   MODEL EXECUTION HELPERS
============================================================ */

async function runModelSafely(
  loader,
  value,
  name
) {

  try {

    const model =
      await loader();


    const score =
      await classify(
        model,
        value
      );


    if (
      !Number.isFinite(
        score
      )
    ) {

      throw new Error(
        `${name} returned invalid score`
      );
    }


    return score;

  } catch (error) {

    console.error(
      `${name} failed:`,
      error
    );


    return NaN;
  }
}


/* ============================================================
   SEGMENT MODEL EXECUTION
============================================================ */

async function analyzeChunksWithTMR(
  model,
  chunks
) {

  const results = [];


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
        ) * 28
      ),

      `Trace Map ${i + 1}/${chunks.length}…`
    );


    try {

      const score =
        await classify(
          model,
          chunks[i]
        );


      results.push(
        score
      );

    } catch (error) {

      console.warn(
        `Segment ${i + 1} failed:`,
        error
      );


      results.push(
        NaN
      );
    }
  }


  return results;
}


/* ============================================================
   MAIN SMART SCAN
============================================================ */

async function runSmartScan() {

  if (
    !textEl
  ) {

    console.error(
      'AI Trace: textarea #text was not found.'
    );

    return;
  }


  const value =
    textEl.value.trim();


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


  const scanButton =
    $('scan');


  if (
    scanButton
  ) {

    scanButton.disabled =
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


    /*
      --------------------------------------------------------
      DOCUMENT PROFILE
      --------------------------------------------------------
    */

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
      calculateHumanEvidence(
        profile,
        domainInfo.domain
      );


    const chunks =
      chunkText(
        value
      );


    /*
      --------------------------------------------------------
      MODEL SCORES
      --------------------------------------------------------
    */

    const scores = {

      tmr:
        NaN,

      e5:
        NaN,

      modern:
        NaN
    };


    let segmentScores = [];

    let thirdUsed = false;


    /*
      --------------------------------------------------------
      MODEL A — TMR
      --------------------------------------------------------
    */

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


      segmentScores =
        await analyzeChunksWithTMR(
          modelA,
          chunks
        );

    } catch (error) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /*
      --------------------------------------------------------
      MODEL B — E5
      --------------------------------------------------------
    */

    setProgress(
      60,
      'Running Model B…'
    );


    scores.e5 =
      await runModelSafely(
        loadE5,
        value,
        'E5-small'
      );


    /*
      Remove failed segment values before routing.
    */

    const usableSegments =
      segmentScores.filter(
        Number.isFinite
      );


    /*
      --------------------------------------------------------
      MODEL C ROUTING
      --------------------------------------------------------
    */

    thirdUsed =
      shouldUseThirdModel({

        scores,

        segmentScores:
          usableSegments,

        domain:
          domainInfo.domain,

        humanScore:
          human.score,

        words,

        language
      });


    /*
      --------------------------------------------------------
      MODEL C — MODERNBERT
      --------------------------------------------------------
    */

    if (
      thirdUsed
    ) {

      setProgress(
        72,
        'Deep verification…'
      );


      scores.modern =
        await runModelSafely(
          loadModern,
          value,
          'ModernBERT'
        );


      if (
        !Number.isFinite(
          scores.modern
        )
      ) {

        thirdUsed =
          false;
      }
    }


    /*
      --------------------------------------------------------
      MODEL FAILURE CHECK
      --------------------------------------------------------
    */

    const activeScores =
      Object.values(
        scores
      )
        .filter(
          Number.isFinite
        );


    if (
      !activeScores.length
    ) {

      throw new Error(
        'No detector completed successfully.'
      );
    }


    /*
      If TMR segment inference completely failed,
      use a neutral segment map.

      This prevents the application from freezing
      or generating NaN values.
    */

    if (
      !usableSegments.length
    ) {

      segmentScores =
        chunks.map(
          () => 50
        );

    } else {

      segmentScores =
        segmentScores.map(
          score =>
            Number.isFinite(
              score
            )
              ? score
              : 50
        );
    }


    /*
      --------------------------------------------------------
      ADAPTIVE BENCHMARK RELIABILITY
      --------------------------------------------------------
    */

    setProgress(
      86,
      'Evaluating detector reliability…'
    );


    const benchmarkRecords =
      loadBench();


    const reliability =
      buildModelReliability(
        domainInfo.domain,
        benchmarkRecords
      );


    /*
      --------------------------------------------------------
      CONSENSUS
      --------------------------------------------------------
    */

    setProgress(
      92,
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

        thirdUsed,

        reliability
      });


    /*
      --------------------------------------------------------
      FINAL SCAN OBJECT
      --------------------------------------------------------
    */

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

      domainScores:
        domainInfo.scores,

      profile,

      scores,

      segmentScores,

      human,

      reliability,

      consensus
    };


    /*
      --------------------------------------------------------
      RENDER

      Isolated so an interface bug cannot make
      successful model inference look frozen.
      --------------------------------------------------------
    */

    try {

      render(
        scan
      );

    } catch (renderError) {

      console.error(
        'Render error:',
        renderError
      );


      setState(
        'Analysis complete • UI error'
      );


      alert(
        'The analysis completed, but the result interface encountered an error. Check the browser console.'
      );
    }


    /*
      --------------------------------------------------------
      HISTORY
      --------------------------------------------------------
    */

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


    /*
      --------------------------------------------------------
      COMPLETE
      --------------------------------------------------------
    */

    setProgress(
      100,
      'Trace complete'
    );


    if (
      isMobileDevice()
    ) {

      setState(
        'V6.2 Mobile Safe • analysis complete ✓'
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V6.2 Adaptive 3-model engine ✓'
      );

    } else {

      setState(
        'V6.2 Adaptive 2-model engine ✓'
      );
    }


    /*
      Benchmark prompt is delayed so the report
      renders first.
    */

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
      700
    );


  } catch (fatalError) {

    console.error(
      'AI Trace fatal scan error:',
      fatalError
    );


    setState(
      'Scan error'
    );


    alert(
      `AI Trace could not complete the analysis.\n\n${fatalError.message || 'Unknown error'}`
    );


  } finally {

    if (
      scanButton
    ) {

      scanButton.disabled =
        false;
    }


    hideProgress();
  }
}
/* ============================================================
   AI TRACE V6.2
   PART 3 / 3

   - Report rendering
   - Adaptive reliability dashboard
   - Benchmark prompt
   - Benchmark metrics
   - Domain metrics
   - Error inspectors
   - History
   - JSON / CSV export
   - Developer utilities
   - Initialization
============================================================ */


/* ============================================================
   MAIN REPORT RENDER
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
    profile,
    reliability
  } = scan;


  $('report')
    ?.classList
    .remove(
      'hidden'
    );


  const resolvedDecision =
    consensus.verdict !==
    'INCONCLUSIVE';


  /* ==========================================================
     TOP SCORE
  ========================================================== */

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
      'INCONCLUSIVE'
    ) {

      explanation =
        `AI Trace abstained because the available evidence was not strong enough for a reliable AI/Human attribution. ` +
        `Diagnostic AI signal: ${consensus.calibrated}%. ` +
        `Evidence sufficiency: ${consensus.sufficiency.score}% (${consensus.sufficiency.level}). ` +
        `Active models: ${consensus.activeModels}/3.`;

    } else {

      explanation =
        `The decision combines adaptive detector reliability, model agreement, domain context, segment stability and human counter-evidence. ` +
        `Diagnostic AI signal: ${consensus.calibrated}%. ` +
        `Evidence sufficiency: ${consensus.sufficiency.score}%.`;
    }


    if (
      consensus.outlier?.detected
    ) {

      explanation +=
        ` ${String(
          consensus.outlier.detector
        ).toUpperCase()} was down-weighted as a possible detector outlier.`;
    }


    $('explain').textContent =
      explanation;
  }


  /* ==========================================================
     TRACE DNA
  ========================================================== */

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


  /* ==========================================================
     ENGINE BADGE
  ========================================================== */

  if ($('engineBadge')) {

    if (
      consensus.activeModels <= 1
    ) {

      $('engineBadge').textContent =
        'V6.2 • LIMITED EVIDENCE';

    } else if (
      consensus.outlier?.detected
    ) {

      $('engineBadge').textContent =
        'V6.2 • ADAPTIVE OUTLIER DEFENSE';

    } else if (
      consensus.thirdUsed
    ) {

      $('engineBadge').textContent =
        'V6.2 • ADAPTIVE 3-MODEL';

    } else if (
      isMobileDevice()
    ) {

      $('engineBadge').textContent =
        'V6.2 • MOBILE SAFE';

    } else {

      $('engineBadge').textContent =
        'V6.2 • ADAPTIVE CONSENSUS';
    }
  }


  /* ==========================================================
     EVIDENCE CONTENT
  ========================================================== */

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


  const reliabilityText =
    detector => {

      const model =
        reliability?.[
          detector
        ];


      if (!model) {
        return 'No reliability data';
      }


      const base =
        model.base ??
        1;


      const ai =
        model.ai?.weight ??
        1;


      const humanWeight =
        model.human?.weight ??
        1;


      return `Base ${Number(base).toFixed(2)} · AI ${Number(ai).toFixed(2)} · Human ${Number(humanWeight).toFixed(2)}`;
    };


  const weightText =
    detector => {

      const item =
        consensus.modelWeights?.[
          detector
        ];


      if (!item) {
        return 'Inactive';
      }


      return `${item.score}% signal · effective weight ${item.weight}`;
    };


  const modernText =
    Number.isFinite(
      scores.modern
    )
      ? `${scores.modern}% AI signal`
      : isMobileDevice()
        ? 'Disabled on mobile for memory stability'
        : 'Not required / unavailable';


  const outlierText =
    consensus.outlier?.detected
      ? `${String(
          consensus.outlier.detector
        ).toUpperCase()} flagged as possible outlier · distance ${consensus.outlier.distance} points`
      : 'No clear detector outlier';


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
      'Adaptive raw signal',
      `${consensus.raw}%`,
      'Weighted ensemble'
    ],

    [
      'Evidence sufficiency',
      `${consensus.sufficiency.score}% — ${consensus.sufficiency.level}`,
      consensus.sufficiency.score >= 75
        ? 'Strong'
        : consensus.sufficiency.score >= 55
          ? 'Moderate'
          : 'Weak'
    ],

    [
      'Model agreement',
      `${consensus.modelAgreement.agreement}%`,
      `Spread ${consensus.modelSpread} pts`
    ],

    [
      'Active models',
      `${consensus.activeModels}/3`,
      consensus.activeModels >= 2
        ? 'Usable redundancy'
        : 'Insufficient'
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
      weightText(
        'tmr'
      ),
      reliabilityText(
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
      weightText(
        'e5'
      ),
      reliabilityText(
        'e5'
      )
    ],

    [
      'ModernBERT judge',
      modernText,
      'Model C'
    ],

    [
      'ModernBERT adaptive weight',
      weightText(
        'modern'
      ),
      reliabilityText(
        'modern'
      )
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
      outlierText,
      consensus.outlier?.detected
        ? 'Down-weighted'
        : 'Clear'
    ],

    [
      'Segment stability',
      `${consensus.segmentAnalysis.stability}%`,
      `Range ${consensus.segmentRange} pts`
    ],

    [
      'Segment profile',
      `${consensus.segmentAnalysis.aiSegments} AI · ${consensus.segmentAnalysis.humanSegments} Human · ${consensus.segmentAnalysis.uncertainSegments} uncertain`,
      consensus.segmentAnalysis.mixed
        ? 'Mixed'
        : 'Consistent'
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
      ${escapeHTML(
        item[0]
      )}
    </span>

    <span>
      ${escapeHTML(
        item[2]
      )}
    </span>

  </div>

  <small>
    ${escapeHTML(
      item[1]
    )}
  </small>

</div>

`
        )
        .join('');
  }


  /* ==========================================================
     DOCUMENT METRICS
  ========================================================== */

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

    'Outlier':
      consensus.outlier?.detected
        ? consensus.outlier.detector
        : 'None',

    Decision:
      consensus.verdict
  };


  if ($('metrics')) {

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

  <span>
    ${escapeHTML(
      key
    )}
  </span>

  <b>
    ${escapeHTML(
      String(
        value
      )
    )}
  </b>

</div>

`
        )
        .join('');
  }


  /* ==========================================================
     TRACE MAP
  ========================================================== */

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
              segmentScores[
                index
              ] ??
              50;


            let label =
              'Uncertain';


            if (
              score >= 70
            ) {

              label =
                'AI-supporting';

            } else if (
              score <= 30
            ) {

              label =
                'Human-supporting';
            }


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

    <i
      style="width:${clamp(
        score
      )}%"
    ></i>

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


  renderBenchmarkPanel();
}


/* ============================================================
   BENCHMARK PROMPT
============================================================ */

function benchmarkPrompt(
  scan
) {

  const answer =
    prompt(
`AI TRACE V6.2 BENCHMARK LAB

Only label samples whose TRUE origin you know.

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


  const prefixes = {

    AI:
      'A',

    HUMAN:
      'H',

    MIXED:
      'M',

    UNKNOWN:
      'U'
  };


  const sameClass =
    records.filter(
      record =>
        record.truth === truth
    ).length;


  const id =
    `${prefixes[
      truth
    ]}-${String(
      sameClass + 1
    ).padStart(
      3,
      '0'
    )}`;


  /*
    IMPORTANT:

    The prediction already happened before this
    ground-truth label is stored.

    Therefore the new sample cannot alter its own
    prediction.
  */

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


  renderBenchmarkPanel();


  const metrics =
    evaluatePredictions(
      binaryRecords(
        records
      ),
      benchmarkPrediction
    );


  alert(
`Benchmark saved: ${id}

AI samples: ${metrics.totalAI}
HUMAN samples: ${metrics.totalHuman}

Coverage: ${metrics.coverage}%
Selective accuracy: ${metrics.selectiveAccuracy}%

False-positive rate: ${metrics.fpr}%
False-negative rate: ${metrics.fnr}%

AI abstention rate: ${metrics.aiAbstainRate}%
Human abstention rate: ${metrics.humanAbstainRate}%

Development measurements only.`
  );
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


  const domains =
    new Map();


  for (
    const record
    of rows
  ) {

    const domain =
      record.domain ||
      'general';


    if (
      !domains.has(
        domain
      )
    ) {

      domains.set(
        domain,
        []
      );
    }


    domains
      .get(
        domain
      )
      .push(
        record
      );
  }


  return [
    ...domains.entries()
  ]
    .map(
      (
        [
          domain,
          domainRows
        ]
      ) => {

        const metrics =
          evaluatePredictions(
            domainRows,
            benchmarkPrediction
          );


        return {

          domain,

          ...metrics
        };
      }
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
   BENCHMARK INSPECTORS
============================================================ */

function falsePositiveRecords(
  records = loadBench()
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
  records = loadBench()
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
  records = loadBench()
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
   BENCHMARK PANEL
============================================================ */

function renderBenchmarkPanel() {

  const report =
    $('report');


  if (!report) {
    return;
  }


  let panel =
    $('benchmarkPanelV62');


  if (!panel) {

    panel =
      document.createElement(
        'section'
      );


    panel.id =
      'benchmarkPanelV62';


    panel.className =
      'panel devPanel';


    panel.style.marginTop =
      '18px';


    report.appendChild(
      panel
    );
  }


  const records =
    loadBench();


  const binary =
    binaryRecords(
      records
    );


  const metrics =
    evaluatePredictions(
      binary,
      benchmarkPrediction
    );


  const readiness =
    benchmarkReadiness(
      records
    );


  const domains =
    domainPerformance(
      records
    );


  const falsePositives =
    falsePositiveRecords(
      records
    );


  const falseNegatives =
    falseNegativeRecords(
      records
    );


  const abstentions =
    abstentionRecords(
      records
    );


  const mixed =
    records.filter(
      record =>
        record.truth ===
        'MIXED'
    );


  const reliability =
    buildModelReliability(
      'general',
      records
    );


  const detectorHTML =
    [
      'tmr',
      'e5',
      'modern'
    ]
      .map(
        detector => {

          const global =
            reliability[
              detector
            ]?.global;


          const metrics =
            global?.metrics;


          const base =
            reliability[
              detector
            ]?.base ??
            1;


          return `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(
        detector.toUpperCase()
      )}
    </span>

    <span>
      ${global?.samples ?? 0} samples
    </span>

  </div>

  <small>
    Reliability ${Number(
      base
    ).toFixed(
      2
    )}
    · Accuracy ${metrics?.selectiveAccuracy ?? 0}%
    · Coverage ${metrics?.coverage ?? 0}%
    · FPR ${metrics?.fpr ?? 0}%
    · FNR ${metrics?.fnr ?? 0}%
  </small>

</div>

`;
        }
      )
      .join('');


  const domainHTML =
    domains
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
    ${escapeHTML(
      record.domain ||
      'general'
    )}
    · Prediction ${escapeHTML(
      benchmarkPrediction(
        record
      )
    )}
    · Signal ${escapeHTML(
      String(
        record.consensus
          ?.calibrated ??
        '?'
      )
    )}%
    · Sufficiency ${escapeHTML(
      String(
        record.consensus
          ?.sufficiency
          ?.score ??
        record.consensus
          ?.quality ??
        '?'
      )
    )}%
  </small>

</div>

`;
  }


  const fpHTML =
    falsePositives
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
    falseNegatives
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


  const abstentionHTML =
    abstentions
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
    records
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


  panel.innerHTML = `

<span class="over">
  V6.2 BENCHMARK LAB • ADAPTIVE RELIABILITY
</span>

<h2>
  Adaptive Reliability Lab
</h2>

<p class="sub">
  Detector weights only begin learning after minimum sample requirements are met. Small benchmark sets remain protected from overfitting.
</p>


<h3>
  Calibration readiness
</h3>

<div class="evidence">

  <div class="ev">

    <div class="evTop">

      <span>
        ${readiness.level}
      </span>

      <span>
        ${
          readiness.level ===
            'COLLECTING'
            ? 'LEARNING DISABLED / LIMITED'
            : 'BENCHMARK LEARNING AVAILABLE'
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

</div>


<h3>
  Ensemble performance
</h3>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${metrics.total}</b>
  </div>

  <div class="metric">
    <span>Coverage</span>
    <b>${metrics.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${metrics.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${metrics.precision}%</b>
  </div>

  <div class="metric">
    <span>AI recall</span>
    <b>${metrics.recall}%</b>
  </div>

  <div class="metric">
    <span>Human specificity</span>
    <b>${metrics.specificity}%</b>
  </div>

  <div class="metric">
    <span>False-positive rate</span>
    <b>${metrics.fpr}%</b>
  </div>

  <div class="metric">
    <span>False-negative rate</span>
    <b>${metrics.fnr}%</b>
  </div>

  <div class="metric">
    <span>AI abstention rate</span>
    <b>${metrics.aiAbstainRate}%</b>
  </div>

  <div class="metric">
    <span>Human abstention rate</span>
    <b>${metrics.humanAbstainRate}%</b>
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
    abstentionHTML ||
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
   JSON EXPORT
============================================================ */

function exportBenchmarkJSON() {

  const records =
    loadBench();


  const data = {

    version:
      VERSION,

    exportedAt:
      nowISO(),

    readiness:
      benchmarkReadiness(
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

    records
  };


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
    `AI-Trace-V62-Benchmark-${Date.now()}.json`;


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

        record.consensus
          ?.sufficiency
          ?.score ??
        record.consensus
          ?.quality,

        record.consensus?.confidence,

        record.consensus?.uncertainty,

        record.consensus?.verdict,

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
    `AI-Trace-V62-Benchmark-${Date.now()}.csv`;


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
   DEVELOPER UTILITIES
============================================================ */

window.AITraceV62 = {

  benchmark() {

    const records =
      loadBench();


    return {

      version:
        VERSION,

      readiness:
        benchmarkReadiness(
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
    domain = 'general'
  ) {

    return buildModelReliability(
      domain,
      loadBench()
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

    const confirmation =
      confirm(
        'Delete all V6.2 benchmark records from this device?'
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
      'V6.2 benchmark data deleted.'
    );
  },


  clearHistory() {

    const confirmation =
      confirm(
        'Delete local V6.2 scan history?'
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
      'V6.2 scan history deleted.'
    );
  }
};


/* ============================================================
   INITIALIZATION
============================================================ */

updateCount();


setTimeout(
  renderBenchmarkPanel,
  350
);


console.info(
  `AI TRACE V${VERSION} loaded`
);
