/*
  ============================================================
  AI TRACE V6.7 — DIAGNOSTIC CALIBRATION ENGINE
  PART 1

  Includes:
  - Core helpers
  - UI helpers
  - Text profiling
  - Storage
  - Device detection
  - Domain-ready document profiling
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

const VERSION = '6.8';


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

/*
  IMPORTANT:

  We deliberately keep the V6.6 benchmark / history / queue keys
  so the existing dataset is not lost during the V6.7 upgrade.
*/

const BENCH_KEY =
  'aiTraceBenchmarkV66';

const HISTORY_KEY =
  'aiTraceHistoryV66';

const QUEUE_KEY =
  'aiTraceCalibrationQueueV66';

const WORKER_STATE_KEY =
  'aiTraceCalibrationWorkerStateV66';


/*
  New V6.7 diagnostic storage.
*/

const DIAGNOSTIC_KEY =
  'aiTraceDiagnosticAuditV67';


/* ============================================================
   LEGACY DATA
============================================================ */

const LEGACY_BENCH_KEYS = [

  'aiTraceBenchmarkV65',

  'aiTraceBenchmarkV64',

  'aiTraceBenchmarkV63',

  'aiTraceBenchmarkV62',

  'aiTraceBenchmarkV61',

  'aiTraceBenchmarkV6',

  'aiTraceBenchmarkV54',

  'aiTraceBenchmarkV53',

  'aiTraceBenchmarkV52',

  'aiTraceBenchmarkV51'

];


const LEGACY_QUEUE_KEYS = [

  'aiTraceCalibrationQueueV64',

  'aiTraceImportedDatasetV63'

];


/* ============================================================
   MODEL CACHE
============================================================ */

let tmrModel =
  null;

let e5Model =
  null;

let modernModel =
  null;


/* ============================================================
   WORKER STATE
============================================================ */

let workerRunning =
  false;

let workerPaused =
  false;

let workerStopRequested =
  false;


/* ============================================================
   FILE IMPORT STATE
============================================================ */

let selectedDatasetFile =
  null;


/* ============================================================
   DOM
============================================================ */

const $ =
  id =>
    document.getElementById(
      id
    );


const textEl =
  $('text');


/* ============================================================
   BASIC HELPERS
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
      value ||
      ''
    )
      .trim();


  if (
    !clean
  ) {

    return 0;
  }


  return clean
    .split(
      /\s+/
    )
    .filter(
      Boolean
    )
    .length;
}


function average(values) {

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
      sum,
      value
    ) =>
      sum +
      value,
    0
  ) /
  usable.length;
}


function median(values) {

  const usable =
    values
      .filter(
        Number.isFinite
      )
      .slice()
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


function percentage(
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


function escapeHTML(value) {

  return String(
    value ??
    ''
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


function detectLanguage(value) {

  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) ||
      []
    ).length;


  const letters =
    (
      value.match(
        /\p{L}/gu
      ) ||
      []
    ).length;


  if (
    !letters
  ) {

    return 'Unknown';
  }


  return (
    latin /
    letters >=
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
    String(
      value
    )
      .match(
        regex
      ) ||
    []
  ).length;
}


/* ============================================================
   DEVICE
============================================================ */

function isMobileDevice() {

  return (

    window
      .matchMedia(
        '(max-width: 768px)'
      )
      .matches ||

    /Android|iPhone|iPad|iPod/i
      .test(
        navigator.userAgent
      )

  );
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
      localStorage
        .getItem(
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

    localStorage
      .setItem(

        key,

        JSON.stringify(
          data
        )

      );


    return true;

  } catch (
    error
  ) {

    console.warn(
      `Could not save ${key}:`,
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


  $('count')
    .textContent =
    `${wordCount(
      textEl?.value ||
      ''
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


  if (
    $('bar')
  ) {

    $('bar')
      .style
      .width =
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


function setState(label) {

  if (
    $('modelState')
  ) {

    $('modelState')
      .textContent =
      label;
  }
}


function setWorkerUI(
  state,
  message,
  percent = null
) {

  if (
    $('batchWorkerState')
  ) {

    $('batchWorkerState')
      .textContent =
      state;
  }


  if (
    $('batchWorkerText')
  ) {

    $('batchWorkerText')
      .textContent =
      message;
  }


  if (
    percent !==
      null &&
    $('batchWorkerBar')
  ) {

    $('batchWorkerBar')
      .style
      .width =
      `${clamp(
        percent
      )}%`;
  }
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
`Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Modern machine learning systems can generate text, summarize documents, analyze images, write computer code, and assist with complex research tasks.

One of the primary advantages of artificial intelligence is its ability to process information at a scale that would be difficult to achieve manually. Organizations can use automated tools to identify patterns, analyze large datasets, improve workflows, and support decision-making.

However, artificial intelligence also creates important challenges. Generated content may contain inaccurate information, misleading claims, or fabricated details. As machine-generated material becomes increasingly convincing, understanding the origin of digital content becomes more difficult.

Reliable AI detection therefore requires multiple sources of evidence, transparent limitations, continuous testing, and careful handling of uncertainty. A responsible detection system should avoid making strong claims when its underlying models disagree.`;


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
      .split(
        /\s+/
      )
      .filter(
        Boolean
      );


  const sentences =
    value
      .split(
        /[.!?]+/
      )
      .map(
        sentence =>
          sentence.trim()
      )
      .filter(
        Boolean
      );


  const paragraphs =
    value
      .split(
        /\n\s*\n/
      )
      .map(
        paragraph =>
          paragraph.trim()
      )
      .filter(
        Boolean
      );


  const lines =
    value
      .split(
        /\n/
      )
      .map(
        line =>
          line.trim()
      )
      .filter(
        Boolean
      );


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
      .filter(
        Boolean
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
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|additionally|nevertheless|in conclusion|as a result|on the other hand)\b/gi
    );


  const subjectiveMarkers =
    countMatches(
      value,
      /\b(I think|I believe|I remember|I feel|perhaps|maybe|in my view|it seems to me)\b/gi
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

    subjectiveMarkers,

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
      (
        profile.quoteCount >= 5 ||
        profile.dialogueLines >= 2
      )
        ? 5
        : 0,


    poetry:
      (
        profile.lines >= 7 &&
        profile.averageLineLength < 65
      )
        ? 5
        : 0,


    academic:
      countMatches(
        content,
        /\b(method|methodology|results|participants|dataset|experiment|analysis|hypothesis|significant|research|abstract|conclusion|findings)\b/g
      ),


    news:
      countMatches(
        content,
        /\b(reuters|reported|officials|government|minister|president|announced|agency|according to|spokesperson)\b/g
      ),


    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|customer|experience)\b/g
      ),


    social:
      countMatches(
        content,
        /\b(imo|lol|reddit|subreddit|tldr|edit:|upvote|downvote|thread)\b/g
      ),


    recipe:
      countMatches(
        content,
        /\b(cup|tablespoon|teaspoon|ingredients|oven|bake|stir|chop|minutes|serve|flour|sugar)\b/g
      ),


    memoir:
      countMatches(
        content,
        /\b(I remember|my childhood|my father|my mother|when I was|years ago|my family|I grew up)\b/g
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

  let score =
    0;


  const reasons =
    [];


  if (
    profile.sentenceBurstiness >=
    0.70
  ) {

    score +=
      20;


    reasons.push(
      'High sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score +=
      12;


    reasons.push(
      'Moderate sentence-length variation'
    );
  }


  if (
    profile.punctuationTypes >=
    5
  ) {

    score +=
      12;


    reasons.push(
      'Rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >=
    3
  ) {

    score +=
      6;
  }


  if (
    profile.firstPerson >=
      4 ||
    profile.subjectiveMarkers >=
      2
  ) {

    score +=
      10;


    reasons.push(
      'Personal or subjective voice'
    );

  } else if (
    profile.firstPerson >
    0
  ) {

    score +=
      5;
  }


  if (
    profile.contractions >=
    3
  ) {

    score +=
      8;


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

    score +=
      14;


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

    score +=
      8;


    reasons.push(
      'Irregular paragraph rhythm'
    );
  }


  if (
    profile.transitions >=
    4
  ) {

    score -=
      6;
  }


  if (
    domain ===
      'books' ||
    domain ===
      'poetry'
  ) {

    score +=
      10;


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

    reasons:
      [
        ...new Set(
          reasons
        )
      ]

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


  const chunks =
    [];


  let current =
    '';


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
    .filter(
      Boolean
    )
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
   V6.7 RAW MODEL OUTPUT NORMALIZATION
============================================================ */

function normalizeRawModelOutput(
  output
) {

  const flattened =
    (
      Array.isArray(
        output
      )
        ? output
        : [
            output
          ]
    )
      .flat(
        Infinity
      );


  return flattened
    .filter(
      item =>
        item &&
        typeof item ===
          'object'
    )
    .map(
      item => {

        const label =
          String(
            item.label ??
            ''
          )
            .trim();


        const score =
          Number(
            item.score
          );


        return {

          label,

          normalizedLabel:
            label
              .toLowerCase()
              .trim(),

          score:
            Number.isFinite(
              score
            )
              ? score
              : null

        };
      }
    )
    .filter(
      item =>
        item.label ||
        item.score !==
          null
    );
}


/* ============================================================
   RAW LABEL FAMILY
============================================================ */

function rawLabelFamily(
  label
) {

  const normalized =
    String(
      label ||
      ''
    )
      .toLowerCase()
      .trim();


  if (
    normalized.includes(
      'human'
    )
  ) {

    return 'EXPLICIT_HUMAN';
  }


  if (
    normalized.includes(
      'ai'
    ) ||
    normalized.includes(
      'machine'
    ) ||
    normalized.includes(
      'generated'
    )
  ) {

    return 'EXPLICIT_AI';
  }


  if (
    normalized ===
    'label_0'
  ) {

    return 'LABEL_0';
  }


  if (
    normalized ===
    'label_1'
  ) {

    return 'LABEL_1';
  }


  if (
    !normalized
  ) {

    return 'EMPTY';
  }


  return 'OTHER';
}


/* ============================================================
   DOMINANT RAW OUTPUT
============================================================ */

function dominantRawOutput(
  rawResults
) {

  const valid =
    (
      rawResults ||
      []
    )
      .filter(
        item =>
          Number.isFinite(
            item?.score
          )
      )
      .slice()
      .sort(
        (
          a,
          b
        ) =>
          b.score -
          a.score
      );


  if (
    !valid.length
  ) {

    return {

      label:
        null,

      normalizedLabel:
        null,

      score:
        null,

      family:
        'EMPTY'

    };
  }


  const top =
    valid[0];


  return {

    label:
      top.label,

    normalizedLabel:
      top.normalizedLabel,

    score:
      top.score,

    family:
      rawLabelFamily(
        top.label
      )

  };
}


/* ============================================================
   CURRENT V6.6 MAPPING
============================================================ */

function aiProbability(
  output,
  detectorName =
    'unknown'
) {

  const results =
    normalizeRawModelOutput(
      output
    );

  const detector =
    String(
      detectorName ||
      'unknown'
    )
      .toLowerCase()
      .trim();


  let ai =
    null;

  let human =
    null;


  for (
    const item
    of results
  ) {

    const label =
      item.normalizedLabel;

    const score =
      item.score;


    if (
      !Number.isFinite(
        score
      )
    ) {

      continue;
    }


    /*
      V6.8 detector-aware mapping.

      Diagnostic audit on the known-origin benchmark established:

      E5-small:
        LABEL_0 -> AI
        LABEL_1 -> HUMAN

      ModernBERT:
        LABEL_0 -> AI
        LABEL_1 -> HUMAN

      TMR:
        Prefer its explicit semantic labels (human / AI / generated / machine).
        Do not force generic LABEL_0/LABEL_1 semantics onto TMR.
    */

    const explicitAI =
      label.includes(
        'ai'
      ) ||
      label.includes(
        'machine'
      ) ||
      label.includes(
        'generated'
      );

    const explicitHuman =
      label.includes(
        'human'
      );


    if (
      explicitAI
    ) {

      ai =
        Math.max(
          ai ??
            0,
          score
        );

      continue;
    }


    if (
      explicitHuman
    ) {

      human =
        Math.max(
          human ??
            0,
          score
        );

      continue;
    }


    if (
      detector ===
        'e5' ||
      detector ===
        'modern'
    ) {

      if (
        label ===
          'label_0'
      ) {

        ai =
          Math.max(
            ai ??
              0,
            score
          );
      }


      if (
        label ===
          'label_1'
      ) {

        human =
          Math.max(
            human ??
              0,
            score
          );
      }
    }
  }


  if (
    ai !==
    null
  ) {

    return clamp(
      ai,
      0,
      1
    );
  }


  if (
    human !==
    null
  ) {

    return clamp(
      1 -
        human,
      0,
      1
    );
  }


  /*
    Unknown detector / unknown label family:
    abstain neutrally instead of guessing from array order.
  */

  return 0.5;
}

/* ============================================================
   DETECTOR VOTE
============================================================ */

function detectorVoteFromProbability(
  probability
) {

  const percent =
    probability *
    100;


  if (
    percent >=
    70
  ) {

    return 'AI';
  }


  if (
    percent <=
    30
  ) {

    return 'HUMAN';
  }


  return 'ABSTAIN';
}


/* ============================================================
   DIAGNOSTIC CLASSIFIER
============================================================ */

async function classifyDiagnostic(
  model,
  value,
  detectorName =
    'unknown'
) {

  const startedAt =
    performance.now();


  try {

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


    const raw =
      normalizeRawModelOutput(
        output
      );


    const dominant =
      dominantRawOutput(
        raw
      );


    const mappedProbability =
      aiProbability(
        output,
        detectorName
      );


    const mappedPercent =
      Math.round(
        mappedProbability *
        100
      );


    return {

      detector:
        detectorName,

      success:
        true,

      raw,

      dominant,

      mappedProbability,

      mappedPercent,

      vote:
        detectorVoteFromProbability(
          mappedProbability
        ),

      latencyMs:
        Math.round(
          performance.now() -
          startedAt
        ),

      error:
        null

    };

  } catch (
    error
  ) {

    console.error(
      `${detectorName} diagnostic failure:`,
      error
    );


    return {

      detector:
        detectorName,

      success:
        false,

      raw:
        [],

      dominant:
        {

          label:
            null,

          normalizedLabel:
            null,

          score:
            null,

          family:
            'EMPTY'

        },

      mappedProbability:
        0.5,

      mappedPercent:
        50,

      vote:
        'ABSTAIN',

      latencyMs:
        Math.round(
          performance.now() -
          startedAt
        ),

      error:
        String(
          error?.message ||
          error
        )

    };
  }
}


/* ============================================================
   BACKWARD COMPATIBLE CLASSIFY
============================================================ */

async function classify(
  model,
  value
) {

  const result =
    await classifyDiagnostic(
      model,
      value,
      'legacy'
    );


  if (
    !result.success
  ) {

    throw new Error(
      result.error ||
      'Detector inference failed.'
    );
  }


  return result
    .mappedPercent;
}


/* ============================================================
   SERIALIZE DIAGNOSTIC RESULT
============================================================ */

function serializeDiagnosticResult(
  result
) {

  if (
    !result
  ) {

    return null;
  }


  return {

    detector:
      result.detector,

    success:
      result.success,

    raw:
      (
        result.raw ||
        []
      ).map(
        item => ({

          label:
            item.label,

          score:
            Number.isFinite(
              item.score
            )
              ? Number(
                  item.score
                    .toFixed(
                      6
                    )
                )
              : null,

          family:
            rawLabelFamily(
              item.label
            )

        })
      ),

    dominant:
      {

        label:
          result.dominant
            ?.label ??
          null,

        score:
          Number.isFinite(
            result.dominant
              ?.score
          )
            ? Number(
                result
                  .dominant
                  .score
                  .toFixed(
                    6
                  )
              )
            : null,

        family:
          result.dominant
            ?.family ||
          'EMPTY'

      },

    mappedAIProbability:
      Number(
        (
          result.mappedProbability ??
          0.5
        ).toFixed(
          6
        )
      ),

    mappedAIPercent:
      result.mappedPercent,

    vote:
      result.vote,

    latencyMs:
      result.latencyMs,

    error:
      result.error

  };
}


/* ============================================================
   MAPPING WARNING
============================================================ */

function diagnosticMappingWarning(
  diagnostic,
  truth =
    null
) {

  if (
    !diagnostic ||
    !diagnostic.success
  ) {

    return {

      warning:
        true,

      type:
        'MODEL_FAILURE',

      message:
        diagnostic?.error ||
        'Detector failed.'

    };
  }


  const family =
    diagnostic
      .dominant
      ?.family;


  const vote =
    diagnostic.vote;


  if (
    family ===
      'EXPLICIT_AI' &&
    vote ===
      'HUMAN'
  ) {

    return {

      warning:
        true,

      type:
        'EXPLICIT_LABEL_CONTRADICTION',

      message:
        'Raw label explicitly indicates AI but mapped vote is HUMAN.'

    };
  }


  if (
    family ===
      'EXPLICIT_HUMAN' &&
    vote ===
      'AI'
  ) {

    return {

      warning:
        true,

      type:
        'EXPLICIT_LABEL_CONTRADICTION',

      message:
        'Raw label explicitly indicates HUMAN but mapped vote is AI.'

    };
  }


  if (
    truth ===
      'AI' &&
    (
      family ===
        'LABEL_0' ||
      family ===
        'LABEL_1'
    ) &&
    vote ===
      'HUMAN'
  ) {

    return {

      warning:
        true,

      type:
        'POSSIBLE_BINARY_LABEL_INVERSION',

      message:
        `${family} dominates known-AI sample but current mapping votes HUMAN.`

    };
  }


  if (
    truth ===
      'HUMAN' &&
    (
      family ===
        'LABEL_0' ||
      family ===
        'LABEL_1'
    ) &&
    vote ===
      'AI'
  ) {

    return {

      warning:
        true,

      type:
        'POSSIBLE_BINARY_LABEL_INVERSION',

      message:
        `${family} dominates known-HUMAN sample but current mapping votes AI.`

    };
  }


  return {

    warning:
      false,

    type:
      'NONE',

    message:
      'No immediate mapping contradiction detected.'

  };
}


/* ============================================================
   BENCHMARK NORMALIZATION
============================================================ */

function normalizeBenchmarkRecord(
  record
) {

  if (
    !record
  ) {

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
      .filter(
        Boolean
      );
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
          .filter(
            Boolean
          );


      saveJSON(
        BENCH_KEY,
        normalized
      );


      console.info(
        `AI Trace V6.7 migrated benchmark from ${legacyKey}`
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
   BINARY RECORDS
============================================================ */

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
   DIAGNOSTIC STORAGE
============================================================ */

function loadDiagnosticAudit() {

  return loadJSON(
    DIAGNOSTIC_KEY,
    {
      version:
        VERSION,

      updatedAt:
        null,

      records:
        []
    }
  );
}


function saveDiagnosticAudit(
  audit
) {

  return saveJSON(
    DIAGNOSTIC_KEY,
    audit
  );
}


function clearDiagnosticAudit() {

  localStorage.removeItem(
    DIAGNOSTIC_KEY
  );
}
/* ============================================================
   CALIBRATION QUEUE LOAD + MIGRATION
============================================================ */

function loadCalibrationQueue() {

  const current =
    loadJSON(
      QUEUE_KEY,
      []
    );

  if (
    Array.isArray(current) &&
    current.length
  ) {
    return current;
  }

  for (
    const legacyKey
    of LEGACY_QUEUE_KEYS
  ) {

    const legacy =
      loadJSON(
        legacyKey,
        []
      );

    if (
      Array.isArray(legacy) &&
      legacy.length
    ) {

      saveJSON(
        QUEUE_KEY,
        legacy
      );

      console.info(
        `AI Trace V6.7 migrated queue from ${legacyKey}`
      );

      return legacy;
    }
  }

  return [];
}


function saveCalibrationQueue(
  queue
) {

  return saveJSON(
    QUEUE_KEY,
    queue
  );
}


/* ============================================================
   DATASET ITEM NORMALIZATION
============================================================ */

function normalizeDatasetItem(
  item,
  index = 0
) {

  if (
    !item ||
    typeof item !== 'object'
  ) {

    throw new Error(
      `Sample ${index + 1}: invalid object`
    );
  }


  const truth =
    String(
      item.truth ||
      item.label ||
      item.groundTruth ||
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
    ].includes(truth)
  ) {

    throw new Error(
      `Sample ${index + 1}: invalid truth label`
    );
  }


  const sampleText =
    String(
      item.text ||
      item.content ||
      ''
    )
      .trim();


  if (
    wordCount(sampleText) <
    30
  ) {

    throw new Error(
      `Sample ${index + 1}: text must contain at least 30 words`
    );
  }


  return {

    truth,

    source:
      String(
        item.source ||
        item.origin ||
        ''
      ).trim(),

    domain:
      String(
        item.domain ||
        'auto'
      )
        .trim()
        .toLowerCase(),

    text:
      sampleText

  };
}


/* ============================================================
   JSONL PARSER
============================================================ */

function parseJSONL(raw) {

  const lines =
    String(
      raw || ''
    )
      .split(/\r?\n/)
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

        samples.push(
          normalizeDatasetItem(
            JSON.parse(line),
            index
          )
        );

      } catch (error) {

        errors.push(
          `Line ${index + 1}: ${error.message}`
        );
      }
    }
  );


  return {

    samples,
    errors,
    format:
      'JSONL'

  };
}


/* ============================================================
   JSON PARSER
============================================================ */

function parseJSONDataset(raw) {

  let parsed;

  try {

    parsed =
      JSON.parse(raw);

  } catch (error) {

    return {

      samples:
        [],

      errors:
        [
          `JSON parse error: ${error.message}`
        ],

      format:
        'JSON'

    };
  }


  let rows = [];


  if (
    Array.isArray(parsed)
  ) {

    rows =
      parsed;

  } else if (
    Array.isArray(parsed.samples)
  ) {

    rows =
      parsed.samples;

  } else if (
    Array.isArray(parsed.records)
  ) {

    rows =
      parsed.records;

  } else if (
    Array.isArray(parsed.data)
  ) {

    rows =
      parsed.data;

  } else {

    return {

      samples:
        [],

      errors:
        [
          'JSON must be an array or contain samples, records or data array.'
        ],

      format:
        'JSON'

    };
  }


  const samples = [];
  const errors = [];


  rows.forEach(
    (
      item,
      index
    ) => {

      try {

        samples.push(
          normalizeDatasetItem(
            item,
            index
          )
        );

      } catch (error) {

        errors.push(
          error.message
        );
      }
    }
  );


  return {

    samples,
    errors,
    format:
      'JSON'

  };
}


/* ============================================================
   TXT PARSER
============================================================ */

function parsePlainTextDataset(raw) {

  /*
    TXT is accepted only when the file
    contains JSONL-compatible data.

    AI Trace does NOT guess ground truth.
  */

  return parseJSONL(
    raw
  );
}


/* ============================================================
   AUTO DATASET PARSER
============================================================ */

function parseDatasetContent(
  raw,
  fileName = ''
) {

  const name =
    String(
      fileName || ''
    )
      .toLowerCase();


  if (
    name.endsWith('.json')
  ) {

    return parseJSONDataset(
      raw
    );
  }


  if (
    name.endsWith('.jsonl') ||
    name.endsWith('.txt')
  ) {

    return parseJSONL(
      raw
    );
  }


  const jsonl =
    parseJSONL(
      raw
    );


  if (
    jsonl.samples.length &&
    !jsonl.errors.length
  ) {

    return jsonl;
  }


  const json =
    parseJSONDataset(
      raw
    );


  if (
    json.samples.length
  ) {

    return json;
  }


  return jsonl;
}


/* ============================================================
   IMPORT STATISTICS
============================================================ */

function importStatistics(
  parsed
) {

  const samples =
    parsed?.samples ||
    [];


  return {

    valid:
      samples.length,

    invalid:
      parsed?.errors
        ?.length ||
      0,

    ai:
      samples.filter(
        item =>
          item.truth === 'AI'
      ).length,

    human:
      samples.filter(
        item =>
          item.truth === 'HUMAN'
      ).length,

    mixed:
      samples.filter(
        item =>
          item.truth === 'MIXED'
      ).length,

    unknown:
      samples.filter(
        item =>
          item.truth === 'UNKNOWN'
      ).length

  };
}


/* ============================================================
   IMPORT PREVIEW
============================================================ */

function renderImportPreview(
  parsed
) {

  const stats =
    importStatistics(
      parsed
    );


  const values = {

    importValidCount:
      stats.valid,

    importInvalidCount:
      stats.invalid,

    importAICount:
      stats.ai,

    importHumanCount:
      stats.human,

    importMixedCount:
      stats.mixed,

    importUnknownCount:
      stats.unknown

  };


  for (
    const [
      id,
      value
    ]
    of Object.entries(values)
  ) {

    if ($(id)) {

      $(id).textContent =
        value;
    }
  }


  if (
    $('bulkImportCount')
  ) {

    $('bulkImportCount')
      .textContent =
      `${stats.valid} samples detected`;
  }
}


/* ============================================================
   FILE INFO
============================================================ */

function renderDatasetFileInfo({
  name = 'No file selected',
  status = 'READY',
  message = 'Select a JSONL, JSON or TXT benchmark dataset.'
} = {}) {

  if (
    !$('datasetFileInfo')
  ) {

    return;
  }


  $('datasetFileInfo')
    .innerHTML = `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(name)}
    </span>

    <span>
      ${escapeHTML(status)}
    </span>

  </div>

  <small>
    ${escapeHTML(message)}
  </small>

</div>

`;
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
      record?.scores?.[
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
   FINAL BENCHMARK PREDICTION
============================================================ */

function benchmarkPrediction(
  record
) {

  const verdict =
    record?.consensus
      ?.verdict ||
    '';


  if (
    verdict === 'Strong AI evidence' ||
    verdict === 'Likely AI'
  ) {

    return 'AI';
  }


  if (
    verdict === 'Strong human evidence' ||
    verdict === 'Likely human'
  ) {

    return 'HUMAN';
  }


  return 'ABSTAIN';
}


/* ============================================================
   METRICS
============================================================ */

function evaluatePredictions(
  rows,
  getPrediction
) {

  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;

  let aiAbstain =
    0;

  let humanAbstain =
    0;


  for (
    const row
    of rows
  ) {

    const prediction =
      getPrediction(
        row
      );


    if (
      prediction ===
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
      prediction === 'AI'
    ) {

      TP++;
    }


    if (
      row.truth === 'HUMAN' &&
      prediction === 'HUMAN'
    ) {

      TN++;
    }


    if (
      row.truth === 'HUMAN' &&
      prediction === 'AI'
    ) {

      FP++;
    }


    if (
      row.truth === 'AI' &&
      prediction === 'HUMAN'
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

    abstentions:
      aiAbstain +
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
   LEAVE ONE OUT
============================================================ */

function leaveOneOutRecords(
  records,
  excludeId = null
) {

  if (
    !excludeId
  ) {

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

  const cleanRows =
    leaveOneOutRecords(
      binaryRecords(records),
      excludeId
    );


  const globalRows =
    cleanRows.filter(
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


  if (
    metrics.fpr >= 40
  ) {

    score *=
      0.75;
  }


  if (
    metrics.fpr >= 70
  ) {

    score *=
      0.60;
  }


  const maturity =
    clamp(
      sampleCount /
      120,
      0,
      1
    );


  const softened =
    1 +
    (
      score -
      1
    ) *
    maturity;


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


  function evaluateDirection(
    rows
  ) {

    let predictions =
      0;

    let correct =
      0;


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
        predicted !==
        direction
      ) {

        continue;
      }


      predictions++;


      if (
        record.truth ===
        direction
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
      stats.globalRows
    );


  const domainSpecific =
    evaluateDirection(
      stats.domainRows
    );


  const globalReady =
    global.predictions >=
    10;


  const domainReady =
    domainSpecific.predictions >=
    6;


  let weight =
    1;


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
      weight *
      0.45 +
      clamp(
        domainSpecific.accuracy /
        100,
        0.20,
        1
      ) *
      0.55;
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
        )
          .toFixed(3)
      ),

    global,

    domain:
      domainSpecific

  };
}


/* ============================================================
   MODEL RELIABILITY
============================================================ */

function buildModelReliability(
  domain,
  records =
    loadBench(),
  excludeId =
    null
) {

  const result =
    {};


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


    result[
      detector
    ] = {

      base:
        calculateReliabilityWeight(
          stats.globalMetrics,
          stats.globalRows.length
        ),

      global: {

        samples:
          stats.globalRows.length,

        metrics:
          stats.globalMetrics

      },

      domain: {

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
   MODEL OUTLIER
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
            score
          ]
        ) =>
          Number.isFinite(
            score
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
            item =>
              item[1]
          )
        )

    };
  }


  const reference =
    median(
      entries.map(
        item =>
          item[1]
      )
    );


  const distances =
    entries
      .map(
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
    first.distance >=
      30 &&
    (
      first.distance -
      second.distance
    ) >=
      14;


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
      deviation *
      1.25
    );


  stability -=
    Math.min(
      35,
      range *
      0.35
    );


  if (
    aiSegments > 0 &&
    humanSegments > 0
  ) {

    stability -=
      10;
  }


  return {

    mean:
      Math.round(mean),

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
        spread *
        1.10 -
        deviation *
        0.50
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

  let score =
    100;


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
    modelAgreement.active ===
    1
  ) {

    score -= 35;

  } else if (
    modelAgreement.active ===
    2
  ) {

    score -= 10;
  }


  score -=
    Math.round(
      (
        100 -
        modelAgreement.agreement
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
   CURRENT MODEL WEIGHT
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


  if (
    !profile
  ) {

    return 1;
  }


  const direction =
    score >= 50
      ? 'ai'
      : 'human';


  let weight =
    profile.base *
      0.68 +
    (
      profile[
        direction
      ]?.weight ??
      1
    ) *
      0.32;


  if (
    readiness.level ===
    'COLLECTING'
  ) {

    weight =
      clamp(
        weight,
        0.90,
        1.08
      );
  }


  if (
    readiness.level ===
    'EXPERIMENTAL'
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
      outlier.detector ===
        detector
    ) {

      weight *=
        0.45;
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
          weight.toFixed(3)
        ),

      outlier:
        outlier.detected &&
        outlier.detector ===
          detector

    };
  }


  let weightedTotal =
    0;


  let totalWeight =
    0;


  for (
    let i = 0;
    i < values.length;
    i++
  ) {

    weightedTotal +=
      values[i] *
      weights[i];


    totalWeight +=
      weights[i];
  }


  const weighted =
    totalWeight
      ? weightedTotal /
        totalWeight
      : 50;


  const robustMedian =
    median(
      values
    );


  const signal =
    clamp(
      Math.round(
        weighted *
          0.62 +
        robustMedian *
          0.38
      )
    );


  return {

    signal,

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
      disagreement *
      0.34
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


  if (
    calibrated >= 95 &&
    (
      modelAgreement.spread > 18 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 80
    )
  ) {

    calibrated =
      94;
  }


  if (
    calibrated <= 5 &&
    (
      modelAgreement.spread > 18 ||
      segmentAnalysis.range > 35 ||
      sufficiency.score < 75
    )
  ) {

    calibrated =
      6;
  }


  let verdict =
    'INCONCLUSIVE';


  const severeConflict =
    modelAgreement.spread >=
    70;


  const highConflict =
    modelAgreement.spread >=
      45 ||
    modelAgreement.agreement <=
      30;


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
    verdict !==
      'INCONCLUSIVE'
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


  const confidenceBase =
    sufficiency.score *
      0.45 +
    modelAgreement.agreement *
      0.35 +
    segmentAnalysis.stability *
      0.20;


  const confidence =
    verdict ===
      'INCONCLUSIVE'
      ? Math.min(
          55,
          Math.round(
            confidenceBase
          )
        )
      : clamp(
          Math.round(
            confidenceBase
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
   V6.7 SINGLE SAMPLE ANALYZER
============================================================ */

async function analyzeSample({
  value,
  truth = null,
  source = '',
  suppliedDomain = null,
  excludeBenchmarkId = null,
  forBatch = false,
  forceAllModels = false
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
    suppliedDomain !==
      'auto'
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


  const diagnostics = {

    tmr:
      null,

    e5:
      null,

    modern:
      null

  };


  let segmentScores =
    [];


  let thirdUsed =
    false;


  /* ========================================================
     TMR FULL DOCUMENT
  ======================================================== */

  try {

    const modelA =
      await loadTMR();


    const diagnostic =
      await classifyDiagnostic(
        modelA,
        value,
        'tmr'
      );


    diagnostics.tmr =
      serializeDiagnosticResult(
        diagnostic
      );


    if (
      diagnostic.success
    ) {

      scores.tmr =
        diagnostic.mappedPercent;

    } else {

      console.warn(
        'TMR diagnostic failed:',
        diagnostic.error
      );
    }


    /* ------------------------------------------------------
       TMR SEGMENT MAP
    ------------------------------------------------------ */

    for (
      const chunk
      of chunks
    ) {

      try {

        const segmentResult =
          await classifyDiagnostic(
            modelA,
            chunk,
            'tmr-segment'
          );


        segmentScores.push(
          segmentResult.success
            ? segmentResult.mappedPercent
            : 50
        );

      } catch (
        error
      ) {

        console.warn(
          'TMR segment failed:',
          error
        );


        segmentScores.push(
          50
        );
      }
    }

  } catch (
    error
  ) {

    console.error(
      'TMR analysis failed:',
      error
    );


    diagnostics.tmr = {

      detector:
        'tmr',

      success:
        false,

      raw:
        [],

      dominant: {

        label:
          null,

        score:
          null,

        family:
          'EMPTY'

      },

      mappedAIProbability:
        0.5,

      mappedAIPercent:
        50,

      vote:
        'ABSTAIN',

      latencyMs:
        0,

      error:
        String(
          error?.message ||
          error
        )

    };
  }


  /* ========================================================
     E5 FULL DOCUMENT
  ======================================================== */

  try {

    const modelB =
      await loadE5();


    const diagnostic =
      await classifyDiagnostic(
        modelB,
        value,
        'e5'
      );


    diagnostics.e5 =
      serializeDiagnosticResult(
        diagnostic
      );


    if (
      diagnostic.success
    ) {

      scores.e5 =
        diagnostic.mappedPercent;

    } else {

      console.warn(
        'E5 diagnostic failed:',
        diagnostic.error
      );
    }

  } catch (
    error
  ) {

    console.error(
      'E5 analysis failed:',
      error
    );


    diagnostics.e5 = {

      detector:
        'e5',

      success:
        false,

      raw:
        [],

      dominant: {

        label:
          null,

        score:
          null,

        family:
          'EMPTY'

      },

      mappedAIProbability:
        0.5,

      mappedAIPercent:
        50,

      vote:
        'ABSTAIN',

      latencyMs:
        0,

      error:
        String(
          error?.message ||
          error
        )

    };
  }


  /* ========================================================
     THIRD MODEL ROUTING
  ======================================================== */

  if (
    forceAllModels
  ) {

    thirdUsed =
      true;

  } else if (
    !isMobileDevice()
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


  /* ========================================================
     MODERNBERT
  ======================================================== */

  if (
    thirdUsed ||
    forceAllModels
  ) {

    try {

      const modelC =
        await loadModern();


      const diagnostic =
        await classifyDiagnostic(
          modelC,
          value,
          'modern'
        );


      diagnostics.modern =
        serializeDiagnosticResult(
          diagnostic
        );


      if (
        diagnostic.success
      ) {

        scores.modern =
          diagnostic.mappedPercent;

      } else {

        console.warn(
          'ModernBERT diagnostic failed:',
          diagnostic.error
        );
      }

    } catch (
      error
    ) {

      console.error(
        'ModernBERT analysis failed:',
        error
      );


      diagnostics.modern = {

        detector:
          'modern',

        success:
          false,

        raw:
          [],

        dominant: {

          label:
            null,

          score:
            null,

          family:
            'EMPTY'

        },

        mappedAIProbability:
          0.5,

        mappedAIPercent:
          50,

        vote:
          'ABSTAIN',

        latencyMs:
          0,

        error:
          String(
            error?.message ||
            error
          )

      };


      if (
        !forceAllModels
      ) {

        thirdUsed =
          false;
      }
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


  /* ========================================================
     DIAGNOSTIC WARNINGS
  ======================================================== */

  const diagnosticWarnings = {

    tmr:
      diagnosticMappingWarning(
        diagnostics.tmr,
        truth
      ),

    e5:
      diagnosticMappingWarning(
        diagnostics.e5,
        truth
      ),

    modern:
      diagnostics.modern
        ? diagnosticMappingWarning(
            diagnostics.modern,
            truth
          )
        : {

            warning:
              false,

            type:
              'NOT_RUN',

            message:
              'ModernBERT was not required for this scan.'

          }

  };


  /* ========================================================
     CONSENSUS
  ======================================================== */

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

    diagnostics,

    diagnosticWarnings,

    segmentScores,

    consensus

  };
}


/* ============================================================
   V6.7 DIAGNOSTIC SAMPLE AUDIT
============================================================ */

async function auditBenchmarkSample(
  record,
  index,
  total
) {

  const text =
    String(
      record?.text ||
      ''
    )
      .trim();


  if (
    wordCount(text) <
    30
  ) {

    return {

      id:
        record?.id ||
        `AUDIT-${index + 1}`,

      truth:
        record?.truth ||
        'UNKNOWN',

      source:
        record?.source ||
        '',

      domain:
        record?.domain ||
        'general',

      success:
        false,

      error:
        'Benchmark record does not contain enough text for re-audit.',

      diagnostics:
        null,

      warnings:
        null

    };
  }


  if (
    $('diagnosticStatusBadge')
  ) {

    $('diagnosticStatusBadge')
      .textContent =
      `AUDITING ${index + 1}/${total}`;
  }


  const analysis =
    await analyzeSample({

      value:
        text,

      truth:
        record.truth,

      source:
        record.source,

      suppliedDomain:
        record.domain,

      excludeBenchmarkId:
        record.id,

      forBatch:
        false,

      /*
        Critical difference from normal Smart Scan:
        diagnostics must inspect ALL detectors.
      */

      forceAllModels:
        true

    });


  return {

    id:
      record.id,

    truth:
      record.truth,

    source:
      record.source,

    domain:
      analysis.domain,

    words:
      analysis.words,

    language:
      analysis.language,

    success:
      true,

    scores:
      analysis.scores,

    diagnostics:
      analysis.diagnostics,

    warnings:
      analysis.diagnosticWarnings,

    consensus:
      {

        calibrated:
          analysis.consensus
            ?.calibrated,

        verdict:
          analysis.consensus
            ?.verdict,

        confidence:
          analysis.consensus
            ?.confidence

      }

  };
}


/* ============================================================
   DETECTOR AUDIT STATISTICS
============================================================ */

function detectorAuditStats(
  auditRecords,
  detector
) {

  const valid =
    auditRecords
      .filter(
        record =>
          record.success &&
          record.diagnostics?.[
            detector
          ]?.success
      );


  const aiRows =
    valid.filter(
      record =>
        record.truth ===
        'AI'
    );


  const humanRows =
    valid.filter(
      record =>
        record.truth ===
        'HUMAN'
    );


  const label0AI =
    aiRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.dominant?.family ===
        'LABEL_0'
    ).length;


  const label1AI =
    aiRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.dominant?.family ===
        'LABEL_1'
    ).length;


  const label0Human =
    humanRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.dominant?.family ===
        'LABEL_0'
    ).length;


  const label1Human =
    humanRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.dominant?.family ===
        'LABEL_1'
    ).length;


  const mappedAIcorrect =
    aiRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.vote ===
        'AI'
    ).length;


  const mappedHumanCorrect =
    humanRows.filter(
      record =>
        record.diagnostics[
          detector
        ]?.vote ===
        'HUMAN'
    ).length;


  const failures =
    auditRecords.filter(
      record =>
        !record.diagnostics?.[
          detector
        ]?.success
    ).length;


  const warningCount =
    valid.filter(
      record =>
        record.warnings?.[
          detector
        ]?.warning
    ).length;


  return {

    detector,

    audited:
      valid.length,

    aiSamples:
      aiRows.length,

    humanSamples:
      humanRows.length,

    failures,

    warnings:
      warningCount,

    label0OnAI:
      label0AI,

    label1OnAI:
      label1AI,

    label0OnHuman:
      label0Human,

    label1OnHuman:
      label1Human,

    mappedAIAccuracy:
      percentage(
        mappedAIcorrect,
        aiRows.length
      ),

    mappedHumanAccuracy:
      percentage(
        mappedHumanCorrect,
        humanRows.length
      )

  };
}


/* ============================================================
   POSSIBLE LABEL INVERSION DETECTION
============================================================ */

function inferBinaryLabelDirection(
  stats
) {

  const enoughAI =
    stats.aiSamples >=
    5;


  const enoughHuman =
    stats.humanSamples >=
    5;


  if (
    !enoughAI ||
    !enoughHuman
  ) {

    return {

      status:
        'INSUFFICIENT_DATA',

      confidence:
        0,

      proposedMapping:
        null,

      message:
        'Need at least 5 known-AI and 5 known-HUMAN audited samples.'

    };
  }


  /*
    Candidate A:
      LABEL_1 = AI
      LABEL_0 = HUMAN
  */

  const normalCorrect =
    stats.label1OnAI +
    stats.label0OnHuman;


  /*
    Candidate B:
      LABEL_0 = AI
      LABEL_1 = HUMAN
  */

  const invertedCorrect =
    stats.label0OnAI +
    stats.label1OnHuman;


  const binaryObservations =
    stats.label0OnAI +
    stats.label1OnAI +
    stats.label0OnHuman +
    stats.label1OnHuman;


  if (
    binaryObservations <
    8
  ) {

    return {

      status:
        'NON_BINARY_OR_INSUFFICIENT',

      confidence:
        0,

      proposedMapping:
        null,

      message:
        'Detector does not provide enough LABEL_0/LABEL_1 observations.'

    };
  }


  const normalRate =
    percentage(
      normalCorrect,
      binaryObservations
    );


  const invertedRate =
    percentage(
      invertedCorrect,
      binaryObservations
    );


  const difference =
    Math.abs(
      normalRate -
      invertedRate
    );


  if (
    invertedRate >=
      70 &&
    invertedRate >=
      normalRate +
      20
  ) {

    return {

      status:
        'POSSIBLE_INVERSION',

      confidence:
        difference,

      proposedMapping: {

        LABEL_0:
          'AI',

        LABEL_1:
          'HUMAN'

      },

      message:
        `Known-origin controls fit LABEL_0=AI / LABEL_1=HUMAN better (${invertedRate}% vs ${normalRate}%).`

    };
  }


  if (
    normalRate >=
      70 &&
    normalRate >=
      invertedRate +
      20
  ) {

    return {

      status:
        'CURRENT_MAPPING_SUPPORTED',

      confidence:
        difference,

      proposedMapping: {

        LABEL_0:
          'HUMAN',

        LABEL_1:
          'AI'

      },

      message:
        `Known-origin controls support the current binary mapping (${normalRate}% vs ${invertedRate}%).`

    };
  }


  return {

    status:
      'AMBIGUOUS',

    confidence:
      difference,

    proposedMapping:
      null,

    message:
      `Binary mapping remains ambiguous (${normalRate}% current vs ${invertedRate}% inverted).`

  };
}


/* ============================================================
   BUILD DIAGNOSTIC AUDIT SUMMARY
============================================================ */

function buildDiagnosticAuditSummary(
  records
) {

  const stats = {

    tmr:
      detectorAuditStats(
        records,
        'tmr'
      ),

    e5:
      detectorAuditStats(
        records,
        'e5'
      ),

    modern:
      detectorAuditStats(
        records,
        'modern'
      )

  };


  const mapping = {

    tmr:
      inferBinaryLabelDirection(
        stats.tmr
      ),

    e5:
      inferBinaryLabelDirection(
        stats.e5
      ),

    modern:
      inferBinaryLabelDirection(
        stats.modern
      )

  };


  const successful =
    records.filter(
      record =>
        record.success
    ).length;


  const failures =
    records.length -
    successful;


  let mappingWarnings =
    0;


  let possibleInversions =
    0;


  let disagreements =
    0;


  for (
    const record
    of records
  ) {

    if (
      !record.success
    ) {

      continue;
    }


    for (
      const detector
      of [
        'tmr',
        'e5',
        'modern'
      ]
    ) {

      if (
        record.warnings?.[
          detector
        ]?.warning
      ) {

        mappingWarnings++;
      }
    }


    const votes =
      [
        record.diagnostics
          ?.tmr
          ?.vote,

        record.diagnostics
          ?.e5
          ?.vote,

        record.diagnostics
          ?.modern
          ?.vote
      ]
        .filter(
          vote =>
            vote &&
            vote !==
              'ABSTAIN'
        );


    if (
      votes.includes('AI') &&
      votes.includes('HUMAN')
    ) {

      disagreements++;
    }
  }


  for (
    const detector
    of [
      'tmr',
      'e5',
      'modern'
    ]
  ) {

    if (
      mapping[
        detector
      ].status ===
      'POSSIBLE_INVERSION'
    ) {

      possibleInversions++;
    }
  }


  return {

    total:
      records.length,

    successful,

    failures,

    coverage:
      percentage(
        successful,
        records.length
      ),

    mappingWarnings,

    possibleInversions,

    disagreements,

    detectorStats:
      stats,

    mapping

  };
}


/* ============================================================
   RUN FULL DIAGNOSTIC AUDIT
============================================================ */

async function runDiagnosticAudit() {

  if (
    workerRunning
  ) {

    alert(
      'Wait for the Calibration Queue to finish before running the Diagnostic Audit.'
    );

    return;
  }


  const benchmark =
    binaryRecords(
      loadBench()
    );


  if (
    !benchmark.length
  ) {

    alert(
      'No known AI/HUMAN benchmark records are available.'
    );

    return;
  }


  const usable =
    benchmark.filter(
      record =>
        wordCount(
          record.text ||
          ''
        ) >= 30
    );


  if (
    !usable.length
  ) {

    alert(
      'The current benchmark records do not contain stored text. Import benchmark samples that include the text field.'
    );

    return;
  }


  const button =
    $('runDiagnosticAudit');


  if (
    button
  ) {

    button.disabled =
      true;

    button.textContent =
      'Audit running…';
  }


  if (
    $('diagnosticStatusBadge')
  ) {

    $('diagnosticStatusBadge')
      .textContent =
      'AUDITING';
  }


  if (
    $('diagnosticConclusion')
  ) {

    $('diagnosticConclusion')
      .innerHTML = `

<div class="ev">

  <div class="evTop">

    <span>
      Diagnostic audit running
    </span>

    <span>
      PLEASE WAIT
    </span>

  </div>

  <small>
    AI Trace is rerunning known-origin samples through all three detectors.
  </small>

</div>

`;
  }


  const auditRecords =
    [];


  try {

    for (
      let index = 0;
      index < usable.length;
      index++
    ) {

      const record =
        usable[
          index
        ];


      try {

        const audited =
          await auditBenchmarkSample(
            record,
            index,
            usable.length
          );


        auditRecords.push(
          audited
        );

      } catch (
        error
      ) {

        console.error(
          'Audit sample failed:',
          error
        );


        auditRecords.push({

          id:
            record.id,

          truth:
            record.truth,

          source:
            record.source,

          domain:
            record.domain,

          success:
            false,

          error:
            String(
              error?.message ||
              error
            ),

          diagnostics:
            null,

          warnings:
            null

        });
      }


      /*
        Let browser update UI between samples.
      */

      await sleep(
        150
      );
    }


    const summary =
      buildDiagnosticAuditSummary(
        auditRecords
      );


    const audit = {

      version:
        VERSION,

      createdAt:
        nowISO(),

      benchmarkSamples:
        benchmark.length,

      auditedSamples:
        usable.length,

      summary,

      records:
        auditRecords

    };


    saveDiagnosticAudit(
      audit
    );


    renderDiagnosticAudit(
      audit
    );


    if (
      $('diagnosticStatusBadge')
    ) {

      $('diagnosticStatusBadge')
        .textContent =
        'AUDIT COMPLETE';
    }


    alert(
      `Diagnostic Audit complete.\n\n${summary.successful}/${summary.total} samples audited.\n${summary.possibleInversions} possible detector mapping inversions found.`
    );

  } catch (
    error
  ) {

    console.error(
      'Diagnostic Audit failed:',
      error
    );


    if (
      $('diagnosticStatusBadge')
    ) {

      $('diagnosticStatusBadge')
        .textContent =
        'AUDIT ERROR';
    }


    alert(
      `Diagnostic Audit failed.\n\n${error?.message || error}`
    );

  } finally {

    if (
      button
    ) {

      button.disabled =
        false;

      button.textContent =
        'Run Diagnostic Audit';
    }
  }
}


/* ============================================================
   DATASET FILE READER
============================================================ */

async function readDatasetFile(
  file
) {

  if (
    !file
  ) {

    throw new Error(
      'No dataset file selected.'
    );
  }


  const maxBytes =
    5 *
    1024 *
    1024;


  if (
    file.size >
    maxBytes
  ) {

    throw new Error(
      'Dataset file is larger than 5 MB.'
    );
  }


  const allowedExtensions = [
    '.jsonl',
    '.json',
    '.txt'
  ];


  const lowerName =
    String(
      file.name ||
      ''
    )
      .toLowerCase();


  if (
    !allowedExtensions
      .some(
        extension =>
          lowerName.endsWith(
            extension
          )
      )
  ) {

    throw new Error(
      'Only .jsonl, .json and .txt files are supported.'
    );
  }


  const content =
    await file.text();


  if (
    !content.trim()
  ) {

    throw new Error(
      'Dataset file is empty.'
    );
  }


  return {

    content,

    parsed:
      parseDatasetContent(
        content,
        file.name
      )

  };
}


/* ============================================================
   SELECT DATASET FILE
============================================================ */

async function handleDatasetFileSelection(
  file
) {

  selectedDatasetFile =
    file ||
    null;


  if (
    !selectedDatasetFile
  ) {

    renderDatasetFileInfo();


    renderImportPreview({

      samples:
        [],

      errors:
        []

    });


    return;
  }


  renderDatasetFileInfo({

    name:
      selectedDatasetFile.name,

    status:
      'READING',

    message:
      `${Math.round(
        selectedDatasetFile.size /
        1024
      )} KB`

  });


  try {

    const {
      content,
      parsed
    } =
      await readDatasetFile(
        selectedDatasetFile
      );


    if (
      $('bulkImportText')
    ) {

      $('bulkImportText')
        .value =
        content;
    }


    renderImportPreview(
      parsed
    );


    renderDatasetFileInfo({

      name:
        selectedDatasetFile.name,

      status:
        parsed.errors.length
          ? 'CHECK'
          : 'VALID',

      message:
        `${parsed.samples.length} valid · ${parsed.errors.length} invalid · ${parsed.format}`

    });


    if (
      $('bulkImportResult')
    ) {

      $('bulkImportResult')
        .classList
        .remove(
          'hidden'
        );


      $('bulkImportResult')
        .innerHTML =
        parsed.errors.length
          ? `

<b>
  ${parsed.samples.length} valid samples.
</b>

<br><br>

${parsed.errors
  .slice(
    0,
    20
  )
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

<b>
  ${parsed.samples.length} valid samples.
</b>

<br>

Dataset file parsed successfully.

`;
    }

  } catch (
    error
  ) {

    console.error(
      'Dataset file error:',
      error
    );


    renderDatasetFileInfo({

      name:
        selectedDatasetFile.name,

      status:
        'ERROR',

      message:
        error.message

    });


    alert(
      error.message
    );
  }
}


/* ============================================================
   CLEAR DATASET FILE
============================================================ */

function clearSelectedDatasetFile() {

  selectedDatasetFile =
    null;


  if (
    $('datasetFileInput')
  ) {

    $('datasetFileInput')
      .value =
      '';
  }


  if (
    $('bulkImportText')
  ) {

    $('bulkImportText')
      .value =
      '';
  }


  renderDatasetFileInfo();


  renderImportPreview({

    samples:
      [],

    errors:
      []

  });


  if (
    $('bulkImportResult')
  ) {

    $('bulkImportResult')
      .classList
      .add(
        'hidden'
      );


    $('bulkImportResult')
      .innerHTML =
      '';
  }
}


/* ============================================================
   BULK VALIDATION
============================================================ */

function validateBulkImport() {

  const raw =
    $('bulkImportText')
      ?.value ||
    '';


  const parsed =
    parseDatasetContent(
      raw,
      selectedDatasetFile
        ?.name ||
      ''
    );


  renderImportPreview(
    parsed
  );


  if (
    $('bulkImportResult')
  ) {

    $('bulkImportResult')
      .classList
      .remove(
        'hidden'
      );


    $('bulkImportResult')
      .innerHTML =
      parsed.errors.length
        ? `

<b>
  ${parsed.samples.length} valid samples.
</b>

<br>

<b>
  ${parsed.errors.length} invalid samples.
</b>

<br><br>

${parsed.errors
  .slice(
    0,
    30
  )
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
        : parsed.samples.length
          ? `

<b>
  ${parsed.samples.length} valid samples.
</b>

<br>

No validation errors detected.

`
          : `

<b>
  No valid samples detected.
</b>

`;
  }


  return parsed;
}


/* ============================================================
   IMPORT INTO CALIBRATION QUEUE
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


  /*
    Also check the benchmark dataset, not just the queue.

    This prevents re-importing a benchmark sample that has already
    been completed and removed/reloaded from a previous session.
  */

  const benchmark =
    loadBench();


  const existingFingerprints =
    new Set();


  for (
    const item
    of queue
  ) {

    existingFingerprints.add(
      `${item.truth}::${item.text}`
    );
  }


  for (
    const item
    of benchmark
  ) {

    if (
      item.text
    ) {

      existingFingerprints.add(
        `${item.truth}::${item.text}`
      );
    }
  }


  const imported =
    [];


  let duplicates =
    0;


  for (
    const sample
    of parsed.samples
  ) {

    const fingerprint =
      `${sample.truth}::${sample.text}`;


    if (
      existingFingerprints.has(
        fingerprint
      )
    ) {

      duplicates++;

      continue;
    }


    const row = {

      importId:
        `IMP-${Date.now()}-${Math.random()
          .toString(36)
          .slice(
            2,
            10
          )}`,

      status:
        'PENDING',

      createdAt:
        nowISO(),

      attempts:
        0,

      truth:
        sample.truth,

      source:
        sample.source,

      domain:
        sample.domain,

      text:
        sample.text

    };


    imported.push(
      row
    );


    existingFingerprints.add(
      fingerprint
    );
  }


  if (
    imported.length
  ) {

    saveCalibrationQueue(
      [
        ...queue,
        ...imported
      ]
    );
  }


  if (
    typeof renderDatasetManager ===
    'function'
  ) {

    renderDatasetManager();
  }


  if (
    $('bulkImportResult')
  ) {

    $('bulkImportResult')
      .classList
      .remove(
        'hidden'
      );


    $('bulkImportResult')
      .innerHTML = `

<b>
  ${imported.length} samples added to the calibration queue.
</b>

<br>

${duplicates} duplicates skipped.

`;
  }


  alert(
    `${imported.length} samples added to the calibration queue.${duplicates ? ` ${duplicates} duplicates skipped.` : ''}`
  );
}


/* ============================================================
   QUEUE SUMMARY
============================================================ */

function queueSummary(
  queue =
    loadCalibrationQueue()
) {

  return {

    total:
      queue.length,

    pending:
      queue.filter(
        row =>
          row.status ===
          'PENDING'
      ).length,

    running:
      queue.filter(
        row =>
          row.status ===
          'RUNNING'
      ).length,

    complete:
      queue.filter(
        row =>
          row.status ===
          'COMPLETE'
      ).length,

    failed:
      queue.filter(
        row =>
          row.status ===
          'FAILED'
      ).length

  };
}


/* ============================================================
   RECOVER INTERRUPTED QUEUE
============================================================ */

function recoverInterruptedQueue() {

  const queue =
    loadCalibrationQueue();


  let changed =
    false;


  for (
    const row
    of queue
  ) {

    if (
      row.status ===
      'RUNNING'
    ) {

      row.status =
        'PENDING';


      row.recoveredAt =
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
   NEXT BENCHMARK ID
============================================================ */

function nextBenchmarkId(
  truth,
  records
) {

  const prefix =
    {

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
    ] ||
    'X';


  const existingNumbers =
    records
      .filter(
        record =>
          String(
            record.id ||
            ''
          )
            .startsWith(
              `${prefix}-`
            )
      )
      .map(
        record =>
          Number(
            String(
              record.id
            )
              .split('-')
              .pop()
          )
      )
      .filter(
        Number.isFinite
      );


  const next =
    existingNumbers.length
      ? Math.max(
          ...existingNumbers
        ) + 1
      : 1;


  return `${prefix}-${String(
    next
  ).padStart(
    4,
    '0'
  )}`;
}
/* ============================================================
   SAVE ANALYZED BATCH RESULT
============================================================ */

function saveAnalyzedBatchRecord(
  queueItem,
  analysis
) {

  const records =
    loadBench();


  const existing =
    records.find(
      record =>
        record.importId ===
        queueItem.importId
    );


  if (
    existing
  ) {

    return existing.id;
  }


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

    /*
      Critical for V6.7:
      Keep benchmark text locally so Diagnostic Audit
      can rerun the known-origin sample through all models.
    */

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
   PROCESS ONE CALIBRATION QUEUE ITEM
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
        item.domain,

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
   WORKER STATE
============================================================ */

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


function saveWorkerState(
  state
) {

  saveJSON(
    WORKER_STATE_KEY,
    state
  );
}


/* ============================================================
   BATCH CALIBRATION WORKER
============================================================ */

async function runCalibrationWorker() {

  if (
    workerRunning
  ) {

    if (
      workerPaused
    ) {

      workerPaused =
        false;


      if (
        $('pauseCalibrationQueue')
      ) {

        $('pauseCalibrationQueue')
          .textContent =
          'Pause';
      }


      setWorkerUI(
        'Running',
        'Calibration worker resumed.'
      );
    }


    return;
  }


  let queue =
    loadCalibrationQueue();


  let summary =
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


  workerStopRequested =
    false;


  if (
    $('runCalibrationQueue')
  ) {

    $('runCalibrationQueue')
      .disabled =
      true;
  }


  if (
    $('pauseCalibrationQueue')
  ) {

    $('pauseCalibrationQueue')
      .textContent =
      'Pause';
  }


  try {

    while (
      true
    ) {

      if (
        workerStopRequested
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
          350
        );


        if (
          workerStopRequested
        ) {

          break;
        }
      }


      if (
        workerStopRequested
      ) {

        break;
      }


      queue =
        loadCalibrationQueue();


      const index =
        queue.findIndex(
          row =>
            row.status ===
            'PENDING'
        );


      if (
        index ===
        -1
      ) {

        break;
      }


      const item = {

        ...queue[
          index
        ]

      };


      queue[
        index
      ].status =
        'RUNNING';


      queue[
        index
      ].startedAt =
        nowISO();


      queue[
        index
      ].attempts =
        (
          queue[
            index
          ].attempts ||
          0
        ) +
        1;


      saveCalibrationQueue(
        queue
      );


      summary =
        queueSummary(
          queue
        );


      const finishedBefore =
        summary.complete +
        summary.failed;


      setWorkerUI(
        'Running',
        `Analyzing ${finishedBefore + 1}/${summary.total} · ${item.truth} · ${wordCount(item.text)} words`,
        summary.total
          ? finishedBefore /
            summary.total *
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
          currentIndex !==
          -1
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
          ) +
          1;


        workerState.lastUpdated =
          nowISO();


        saveWorkerState(
          workerState
        );

      } catch (
        error
      ) {

        console.error(
          'Calibration sample failed:',
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
          currentIndex !==
          -1
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
          ) +
          1;


        workerState.lastUpdated =
          nowISO();


        saveWorkerState(
          workerState
        );
      }


      summary =
        queueSummary(
          loadCalibrationQueue()
        );


      const processed =
        summary.complete +
        summary.failed;


      setWorkerUI(
        'Running',
        `${processed}/${summary.total} processed · ${summary.failed} failed`,
        summary.total
          ? processed /
            summary.total *
            100
          : 0
      );


      if (
        typeof renderDatasetManager ===
        'function'
      ) {

        renderDatasetManager();
      }


      if (
        typeof renderCalibrationLab ===
        'function'
      ) {

        renderCalibrationLab();
      }


      /*
        Give the browser some time to repaint and prevent
        the Dataset Manager from appearing frozen.
      */

      await sleep(
        300
      );
    }


    const finalSummary =
      queueSummary(
        loadCalibrationQueue()
      );


    if (
      workerStopRequested
    ) {

      setWorkerUI(
        'Stopped',
        `${finalSummary.pending} pending · ${finalSummary.complete} complete · ${finalSummary.failed} failed`,
        finalSummary.total
          ? (
              finalSummary.complete +
              finalSummary.failed
            ) /
            finalSummary.total *
            100
          : 0
      );

    } else {

      setWorkerUI(
        finalSummary.failed
          ? 'Complete with errors'
          : 'Complete',

        `${finalSummary.complete} complete · ${finalSummary.failed} failed`,

        100
      );
    }

  } finally {

    workerRunning =
      false;


    workerPaused =
      false;


    if (
      $('runCalibrationQueue')
    ) {

      $('runCalibrationQueue')
        .disabled =
        false;
    }


    if (
      $('pauseCalibrationQueue')
    ) {

      $('pauseCalibrationQueue')
        .textContent =
        'Pause';
    }


    if (
      typeof renderDatasetManager ===
      'function'
    ) {

      renderDatasetManager();
    }


    if (
      typeof renderCalibrationLab ===
      'function'
    ) {

      renderCalibrationLab();
    }
  }
}


/* ============================================================
   PAUSE / RESUME CALIBRATION WORKER
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
    $('pauseCalibrationQueue')
  ) {

    $('pauseCalibrationQueue')
      .textContent =
      workerPaused
        ? 'Resume'
        : 'Pause';
  }


  setWorkerUI(
    workerPaused
      ? 'Paused'
      : 'Running',

    workerPaused
      ? 'Worker will pause before the next sample.'
      : 'Calibration worker resumed.'
  );
}


/* ============================================================
   V6.7 CURRENT SCAN RAW DIAGNOSTICS
============================================================ */

function renderScanRawDiagnostics(
  scan
) {

  const container =
    $('scanRawDiagnostics');


  if (
    !container
  ) {

    return;
  }


  const diagnostics =
    scan?.diagnostics ||
    {};


  const warnings =
    scan?.diagnosticWarnings ||
    {};


  const detectorNames = {

    tmr:
      'TMR',

    e5:
      'E5-small',

    modern:
      'ModernBERT'

  };


  const cards =
    [];


  for (
    const detector
    of [
      'tmr',
      'e5',
      'modern'
    ]
  ) {

    const result =
      diagnostics[
        detector
      ];


    const warning =
      warnings[
        detector
      ];


    if (
      !result
    ) {

      cards.push(`

<div class="ev">

  <div class="evTop">

    <span>
      ${detectorNames[detector]}
    </span>

    <span>
      NOT USED
    </span>

  </div>

  <small>
    Detector was not required for this Smart Scan.
  </small>

</div>

`);

      continue;
    }


    if (
      !result.success
    ) {

      cards.push(`

<div class="ev">

  <div class="evTop">

    <span>
      ${detectorNames[detector]}
    </span>

    <span>
      FAILURE
    </span>

  </div>

  <small>
    ${escapeHTML(
      result.error ||
      'Model inference failed.'
    )}
  </small>

</div>

`);

      continue;
    }


    const dominantLabel =
      result.dominant
        ?.label ||
      'Unknown';


    const dominantScore =
      Number.isFinite(
        Number(
          result.dominant
            ?.score
        )
      )
        ? `${(
            Number(
              result.dominant.score
            ) *
            100
          ).toFixed(2)}%`
        : 'N/A';


    const rawLine =
      (
        result.raw ||
        []
      )
        .map(
          item => {

            const rawPercent =
              Number.isFinite(
                Number(
                  item.score
                )
              )
                ? (
                    Number(
                      item.score
                    ) *
                    100
                  ).toFixed(
                    2
                  )
                : 'N/A';


            return `${item.label}: ${rawPercent}%`;
          }
        )
        .join(
          ' · '
        ) ||
      'No raw output';


    cards.push(`

<div class="ev">

  <div class="evTop">

    <span>
      ${detectorNames[detector]}
    </span>

    <span>
      ${escapeHTML(
        result.vote ||
        'ABSTAIN'
      )}
    </span>

  </div>

  <small>

    <b>Raw:</b>
    ${escapeHTML(rawLine)}

    <br>

    <b>Dominant:</b>
    ${escapeHTML(dominantLabel)}
    (${escapeHTML(dominantScore)})

    <br>

    <b>Mapped AI probability:</b>
    ${escapeHTML(
      String(
        result.mappedAIPercent ??
        50
      )
    )}%

    <br>

    <b>Label family:</b>
    ${escapeHTML(
      result.dominant
        ?.family ||
      'UNKNOWN'
    )}

    <br>

    <b>Latency:</b>
    ${escapeHTML(
      String(
        result.latencyMs ??
        0
      )
    )} ms

    ${
      warning?.warning
        ? `

          <br>

          <b>⚠ Diagnostic warning:</b>
          ${escapeHTML(
            warning.message ||
            warning.type
          )}

        `
        : ''
    }

  </small>

</div>

`);
  }


  container.innerHTML =
    cards.join(
      ''
    );


  if (
    $('scanDiagnosticStatus')
  ) {

    const warningCount =
      Object.values(
        warnings
      )
        .filter(
          warning =>
            warning?.warning
        )
        .length;


    $('scanDiagnosticStatus')
      .textContent =
      warningCount
        ? `${warningCount} WARNING${warningCount === 1 ? '' : 'S'}`
        : 'RAW OUTPUT CAPTURED';
  }
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

    $('scan')
      .disabled =
      true;
  }


  try {

    setState(
      'V6.7 analyzing…'
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
      94,
      'Building evidence report…'
    );


    renderScan(
      analysis
    );


    /*
      New V6.7 panel.
    */

    renderScanRawDiagnostics(
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
        'V6.7 Mobile Safe ✓'
      );

    } else if (
      analysis.consensus
        .thirdUsed
    ) {

      setState(
        'V6.7 Diagnostic 3-model engine ✓'
      );

    } else {

      setState(
        'V6.7 Diagnostic engine ✓'
      );
    }


    /*
      Existing benchmark prompt remains enabled.
    */

    setTimeout(
      () => {

        try {

          if (
            typeof benchmarkPrompt ===
            'function'
          ) {

            benchmarkPrompt(
              analysis
            );
          }

        } catch (
          error
        ) {

          console.warn(
            'Benchmark prompt failed:',
            error
          );
        }

      },
      700
    );

  } catch (
    error
  ) {

    console.error(
      'Smart scan failed:',
      error
    );


    setState(
      'Scan error'
    );


    if (
      $('scanDiagnosticStatus')
    ) {

      $('scanDiagnosticStatus')
        .textContent =
        'ERROR';
    }


    alert(
      `AI Trace could not complete the scan.\n\n${error?.message || 'Unknown error'}`
    );

  } finally {

    if (
      $('scan')
    ) {

      $('scan')
        .disabled =
        false;
    }


    hideProgress();
  }
}
/* ============================================================
   V6.7 SCAN REPORT RENDER
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


  /* ========================================================
     SCORE
  ======================================================== */

  if (
    $('score')
  ) {

    $('score')
      .textContent =
      resolved
        ? `${consensus.calibrated}%`
        : '—';
  }


  if (
    $('scaleFill')
  ) {

    $('scaleFill')
      .style
      .width =
      resolved
        ? `${consensus.calibrated}%`
        : '0%';
  }


  if (
    $('verdict')
  ) {

    $('verdict')
      .textContent =
      consensus.verdict;
  }


  /* ========================================================
     CONFIDENCE
  ======================================================== */

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

    $('confidence')
      .textContent =
      `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;
  }


  /* ========================================================
     EXPLANATION
  ======================================================== */

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


    $('explain')
      .textContent =
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


  if (
    $('humanVal')
  ) {

    $('humanVal')
      .textContent =
      `${humanDisplay}%`;
  }


  if (
    $('aiVal')
  ) {

    $('aiVal')
      .textContent =
      resolved
        ? `${consensus.calibrated}%`
        : 'N/A';
  }


  if (
    $('uncertainVal')
  ) {

    $('uncertainVal')
      .textContent =
      `${consensus.uncertainty}%`;
  }


  if (
    $('humanBar')
  ) {

    $('humanBar')
      .style
      .width =
      `${humanDisplay}%`;
  }


  if (
    $('aiBar')
  ) {

    $('aiBar')
      .style
      .width =
      resolved
        ? `${consensus.calibrated}%`
        : '0%';
  }


  if (
    $('uncertainBar')
  ) {

    $('uncertainBar')
      .style
      .width =
      `${consensus.uncertainty}%`;
  }


  /* ========================================================
     ENGINE BADGE
  ======================================================== */

  if (
    $('engineBadge')
  ) {

    if (
      consensus.outlier.detected
    ) {

      $('engineBadge')
        .textContent =
        'V6.7 • DIAGNOSTIC OUTLIER DEFENSE';

    } else if (
      consensus.thirdUsed
    ) {

      $('engineBadge')
        .textContent =
        'V6.7 • DIAGNOSTIC 3-MODEL';

    } else {

      $('engineBadge')
        .textContent =
        'V6.7 • DIAGNOSTIC CONSENSUS';
    }
  }


  /* ========================================================
     MODEL WEIGHT HELPERS
  ======================================================== */

  const modelWeightLine =
    detector => {

      const item =
        consensus.modelWeights?.[
          detector
        ];


      if (
        !item
      ) {

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


      if (
        !item
      ) {

        return 'No reliability data';
      }


      return (
        `Base ${Number(
          item.base
        ).toFixed(2)} · ` +

        `AI ${Number(
          item.ai?.weight ??
          1
        ).toFixed(2)} · ` +

        `Human ${Number(
          item.human?.weight ??
          1
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


  /* ========================================================
     EVIDENCE CARDS
  ======================================================== */

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
      'E5-small detector',
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
      'ModernBERT judge',
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

    $('evidence')
      .innerHTML =
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


  /* ========================================================
     DOCUMENT METRICS
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

    $('metrics')
      .innerHTML =
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


  /* ========================================================
     TRACE MAP
  ======================================================== */

  const sourceText =
    textEl?.value
      ?.trim() ||
    scan.text ||
    '';


  const chunks =
    chunkText(
      sourceText
    );


  if (
    $('segments')
  ) {

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

  /*
    Do not force benchmark labeling after every scan
    during diagnostic development.

    User can Cancel to ignore the scan.
  */

  const answer =
    prompt(
`AI TRACE V6.7 BENCHMARK

Only label samples whose TRUE origin you know.

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


  const id =
    nextBenchmarkId(
      truth,
      records
    );


  /*
    Preserve the original scanned text.

    This is essential because V6.7 Diagnostic Audit
    needs to rerun known-origin samples.
  */

  const originalText =
    textEl?.value
      ?.trim() ||
    '';


  records.push({

    id,

    truth,

    source,

    savedAt:
      nowISO(),

    predictionFrozen:
      true,

    ...scan,

    text:
      originalText

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

      $(id)
        .textContent =
        value;
    }
  }


  if (
    $('datasetStatusBadge')
  ) {

    $('datasetStatusBadge')
      .textContent =
      readiness.level;
  }


  if (
    $('benchmarkReadinessTop')
  ) {

    $('benchmarkReadinessTop')
      .textContent =
      `Dataset: ${readiness.level}`;
  }


  /* ========================================================
     RECENT RECORDS
  ======================================================== */

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


    $('datasetRecords')
      .innerHTML =
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

    ·

    ${escapeHTML(
      record.source ||
      'No source'
    )}

    ·

    ${
      record.words ||
      wordCount(
        record.text ||
        ''
      ) ||
      0
    } words

  </small>

</div>

`
            )
            .join('')
        : `

<div class="ev">

  <small>
    No benchmark records yet.
  </small>

</div>

`;
  }


  /* ========================================================
     QUEUE STATE
  ======================================================== */

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
        : queue.total
          ? queue.failed
            ? 'Complete with errors'
            : 'Complete'
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
   FALSE POSITIVES
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


/* ============================================================
   FALSE NEGATIVES
============================================================ */

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


/* ============================================================
   ABSTENTIONS
============================================================ */

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

    ${escapeHTML(
      record.domain ||
      'general'
    )}

    · Prediction

    ${escapeHTML(
      benchmarkPrediction(
        record
      )
    )}

    · Signal

    ${
      record.consensus
        ?.calibrated ??
      '?'
    }%

    · Sufficiency

    ${
      record.consensus
        ?.sufficiency
        ?.score ??
      '?'
    }%

  </small>

</div>

`
      )
      .join('');
}
/* ============================================================
   V6.7 DIAGNOSTIC AUDIT RENDER
============================================================ */

function renderDiagnosticAudit(
  audit =
    loadDiagnosticAudit()
) {

  if (
    !audit ||
    !Array.isArray(
      audit.records
    ) ||
    !audit.records.length
  ) {

    if (
      $('diagnosticStatusBadge')
    ) {

      $('diagnosticStatusBadge')
        .textContent =
        'NOT AUDITED';
    }


    return;
  }


  const records =
    audit.records;


  const summary =
    audit.summary ||
    buildDiagnosticAuditSummary(
      records
    );


  /* ========================================================
     STATUS BADGE
  ======================================================== */

  if (
    $('diagnosticStatusBadge')
  ) {

    $('diagnosticStatusBadge')
      .textContent =
      summary.possibleInversions
        ? 'MAPPING WARNING'
        : 'AUDIT COMPLETE';
  }


  /* ========================================================
     DIAGNOSTIC SUMMARY
  ======================================================== */

  const summaryValues = {

    diagnosticSampleCount:
      summary.total,

    diagnosticMappingWarnings:
      summary.mappingWarnings,

    diagnosticModelFailures:
      summary.failures,

    diagnosticPossibleInversions:
      summary.possibleInversions,

    diagnosticDisagreements:
      summary.disagreements,

    diagnosticCoverage:
      `${summary.coverage}%`

  };


  for (
    const [
      id,
      value
    ]
    of Object.entries(
      summaryValues
    )
  ) {

    if (
      $(id)
    ) {

      $(id)
        .textContent =
        value;
    }
  }


  /* ========================================================
     LABEL MAPPING INTEGRITY
  ======================================================== */

  if (
    $('labelMappingStatus')
  ) {

    const detectorNames = {

      tmr:
        'TMR',

      e5:
        'E5-small',

      modern:
        'ModernBERT'

    };


    $('labelMappingStatus')
      .innerHTML =
      [
        'tmr',
        'e5',
        'modern'
      ]
        .map(
          detector => {

            const mapping =
              summary.mapping?.[
                detector
              ];


            if (
              !mapping
            ) {

              return '';
            }


            const proposed =
              mapping.proposedMapping
                ? `LABEL_0=${mapping.proposedMapping.LABEL_0} · LABEL_1=${mapping.proposedMapping.LABEL_1}`
                : 'No mapping decision';


            return `

<div class="ev">

  <div class="evTop">

    <span>
      ${detectorNames[detector]}
    </span>

    <span>
      ${escapeHTML(
        mapping.status
      )}
    </span>

  </div>

  <small>

    ${escapeHTML(
      mapping.message
    )}

    <br>

    <b>
      Mapping:
    </b>

    ${escapeHTML(
      proposed
    )}

    <br>

    <b>
      Direction confidence:
    </b>

    ${escapeHTML(
      String(
        mapping.confidence ??
        0
      )
    )}%

  </small>

</div>

`;
          }
        )
        .join('');
  }


  /* ========================================================
     DETECTOR RAW OUTPUT INTEGRITY
  ======================================================== */

  if (
    $('detectorRawDiagnostics')
  ) {

    const detectorNames = {

      tmr:
        'TMR',

      e5:
        'E5-small',

      modern:
        'ModernBERT'

    };


    $('detectorRawDiagnostics')
      .innerHTML =
      [
        'tmr',
        'e5',
        'modern'
      ]
        .map(
          detector => {

            const stats =
              summary.detectorStats?.[
                detector
              ];


            if (
              !stats
            ) {

              return '';
            }


            return `

<div class="ev">

  <div class="evTop">

    <span>
      ${detectorNames[detector]}
    </span>

    <span>
      ${stats.audited} audited
    </span>

  </div>

  <small>

    <b>Known AI:</b>
    ${stats.aiSamples}

    ·

    <b>Known HUMAN:</b>
    ${stats.humanSamples}

    <br>

    <b>LABEL_0 on AI:</b>
    ${stats.label0OnAI}

    ·

    <b>LABEL_1 on AI:</b>
    ${stats.label1OnAI}

    <br>

    <b>LABEL_0 on HUMAN:</b>
    ${stats.label0OnHuman}

    ·

    <b>LABEL_1 on HUMAN:</b>
    ${stats.label1OnHuman}

    <br>

    <b>Mapped AI correctness:</b>
    ${stats.mappedAIAccuracy}%

    ·

    <b>Mapped HUMAN correctness:</b>
    ${stats.mappedHumanAccuracy}%

    <br>

    <b>Warnings:</b>
    ${stats.warnings}

    ·

    <b>Failures:</b>
    ${stats.failures}

  </small>

</div>

`;
          }
        )
        .join('');
  }


  /* ========================================================
     KNOWN AI CONTROLS
  ======================================================== */

  if (
    $('diagnosticAIControls')
  ) {

    const aiRecords =
      records.filter(
        record =>
          record.truth ===
          'AI'
      );


    $('diagnosticAIControls')
      .innerHTML =
      aiRecords.length
        ? aiRecords
            .slice(
              0,
              20
            )
            .map(
              record => {

                const tmr =
                  record.diagnostics
                    ?.tmr;


                const e5 =
                  record.diagnostics
                    ?.e5;


                const modern =
                  record.diagnostics
                    ?.modern;


                return `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(
        record.id ||
        'AI sample'
      )}
    </span>

    <span>
      KNOWN AI
    </span>

  </div>

  <small>

    TMR:
    ${escapeHTML(
      tmr?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      tmr?.vote ||
      'N/A'
    )}

    (${tmr?.mappedAIPercent ?? '?'}%)

    <br>

    E5:
    ${escapeHTML(
      e5?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      e5?.vote ||
      'N/A'
    )}

    (${e5?.mappedAIPercent ?? '?'}%)

    <br>

    Modern:
    ${escapeHTML(
      modern?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      modern?.vote ||
      'N/A'
    )}

    (${modern?.mappedAIPercent ?? '?'}%)

  </small>

</div>

`;
              }
            )
            .join('')
        : `

<div class="ev">

  <small>
    No known-AI diagnostic samples available.
  </small>

</div>

`;
  }


  /* ========================================================
     KNOWN HUMAN CONTROLS
  ======================================================== */

  if (
    $('diagnosticHumanControls')
  ) {

    const humanRecords =
      records.filter(
        record =>
          record.truth ===
          'HUMAN'
      );


    $('diagnosticHumanControls')
      .innerHTML =
      humanRecords.length
        ? humanRecords
            .slice(
              0,
              20
            )
            .map(
              record => {

                const tmr =
                  record.diagnostics
                    ?.tmr;


                const e5 =
                  record.diagnostics
                    ?.e5;


                const modern =
                  record.diagnostics
                    ?.modern;


                return `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(
        record.id ||
        'HUMAN sample'
      )}
    </span>

    <span>
      KNOWN HUMAN
    </span>

  </div>

  <small>

    TMR:
    ${escapeHTML(
      tmr?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      tmr?.vote ||
      'N/A'
    )}

    (${tmr?.mappedAIPercent ?? '?'}%)

    <br>

    E5:
    ${escapeHTML(
      e5?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      e5?.vote ||
      'N/A'
    )}

    (${e5?.mappedAIPercent ?? '?'}%)

    <br>

    Modern:
    ${escapeHTML(
      modern?.dominant
        ?.label ||
      'N/A'
    )}

    →

    ${escapeHTML(
      modern?.vote ||
      'N/A'
    )}

    (${modern?.mappedAIPercent ?? '?'}%)

  </small>

</div>

`;
              }
            )
            .join('')
        : `

<div class="ev">

  <small>
    No known-HUMAN diagnostic samples available.
  </small>

</div>

`;
  }


  /* ========================================================
     RAW RECORD INSPECTOR
  ======================================================== */

  if (
    $('diagnosticRecords')
  ) {

    $('diagnosticRecords')
      .innerHTML =
      records
        .slice()
        .reverse()
        .slice(
          0,
          40
        )
        .map(
          record => {

            if (
              !record.success
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
      AUDIT FAILURE
    </span>

  </div>

  <small>
    ${escapeHTML(
      record.error ||
      'Unknown error'
    )}
  </small>

</div>

`;
            }


            const detectorLine =
              detector => {

                const diagnostic =
                  record.diagnostics?.[
                    detector
                  ];


                if (
                  !diagnostic
                ) {

                  return `${detector.toUpperCase()}: N/A`;
                }


                const label =
                  diagnostic.dominant
                    ?.label ||
                  'Unknown';


                const rawScore =
                  Number.isFinite(
                    Number(
                      diagnostic.dominant
                        ?.score
                    )
                  )
                    ? `${(
                        Number(
                          diagnostic.dominant.score
                        ) *
                        100
                      ).toFixed(2)}%`
                    : 'N/A';


                return (
                  `${detector.toUpperCase()}: ` +
                  `${label} ${rawScore} → ` +
                  `${diagnostic.mappedAIPercent}% AI → ` +
                  `${diagnostic.vote}`
                );
              };


            const warningList =
              [
                'tmr',
                'e5',
                'modern'
              ]
                .map(
                  detector =>
                    record.warnings?.[
                      detector
                    ]
                )
                .filter(
                  warning =>
                    warning?.warning
                )
                .map(
                  warning =>
                    warning.message
                );


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
        'UNKNOWN'
      )}
    </span>

  </div>

  <small>

    ${escapeHTML(
      detectorLine(
        'tmr'
      )
    )}

    <br>

    ${escapeHTML(
      detectorLine(
        'e5'
      )
    )}

    <br>

    ${escapeHTML(
      detectorLine(
        'modern'
      )
    )}

    ${
      warningList.length
        ? `

        <br>

        <b>
          ⚠ Warnings:
        </b>

        ${escapeHTML(
          warningList.join(
            ' | '
          )
        )}

        `
        : ''
    }

  </small>

</div>

`;
          }
        )
        .join('');
  }


  /* ========================================================
     DIAGNOSTIC CONCLUSION
  ======================================================== */

  if (
    $('diagnosticConclusion')
  ) {

    const inversionDetectors =
      Object.entries(
        summary.mapping ||
        {}
      )
        .filter(
          (
            [
              ,
              mapping
            ]
          ) =>
            mapping.status ===
            'POSSIBLE_INVERSION'
        )
        .map(
          (
            [
              detector
            ]
          ) =>
            detector.toUpperCase()
        );


    let title =
      'No confirmed mapping defect';


    let status =
      'REVIEW';


    let message =
      'The current audit does not provide strong enough evidence to change label mappings automatically.';


    if (
      inversionDetectors.length
    ) {

      title =
        'Possible label inversion detected';


      status =
        'ACTION REQUIRED';


      message =
        `${inversionDetectors.join(', ')} show known-origin behavior more consistent with an inverted LABEL_0/LABEL_1 mapping. Do not change the mapping automatically until HUMAN controls are sufficiently represented.`;

    } else if (
      summary.total >= 20 &&
      summary.coverage >= 90
    ) {

      title =
        'Diagnostic audit completed';


      status =
        'NO AUTO CHANGE';


      message =
        'Raw outputs were captured successfully. Current mappings remain unchanged until a balanced known-AI / known-HUMAN benchmark supports a mapping decision.';
    }


    $('diagnosticConclusion')
      .innerHTML = `

<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(
        title
      )}
    </span>

    <span>
      ${escapeHTML(
        status
      )}
    </span>

  </div>

  <small>
    ${escapeHTML(
      message
    )}
  </small>

</div>

`;
  }
}


/* ============================================================
   V6.7 CALIBRATION LAB
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

    $('calibrationStatusBadge')
      .textContent =
      readiness.level;
  }


  /* ========================================================
     READINESS
  ======================================================== */

  if (
    $('calibrationReadiness')
  ) {

    $('calibrationReadiness')
      .innerHTML = `

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


  /* ========================================================
     ENSEMBLE METRICS
  ======================================================== */

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


    $('ensembleMetrics')
      .innerHTML =
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


  /* ========================================================
     DETECTOR RELIABILITY
  ======================================================== */

  if (
    $('detectorReliability')
  ) {

    const reliability =
      buildModelReliability(
        'general',
        records
      );


    $('detectorReliability')
      .innerHTML =
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

    Base reliability
    ${Number(
      item.base
    ).toFixed(2)}

    · AI weight
    ${Number(
      item.ai.weight
    ).toFixed(2)}

    · Human weight
    ${Number(
      item.human.weight
    ).toFixed(2)}

    · Accuracy
    ${metrics.selectiveAccuracy}%

    · Coverage
    ${metrics.coverage}%

    · FPR
    ${metrics.fpr}%

    · FNR
    ${metrics.fnr}%

  </small>

</div>

`;
          }
        )
        .join('');
  }


  /* ========================================================
     DOMAIN PERFORMANCE
  ======================================================== */

  if (
    $('domainPerformance')
  ) {

    const domains =
      domainPerformance(
        records
      );


    $('domainPerformance')
      .innerHTML =
      domains.length
        ? domains
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

  <span>
    No domain data
  </span>

  <b>
    —
  </b>

</div>

`;
  }


  /* ========================================================
     INSPECTORS
  ======================================================== */

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
   LOAD STORED DIAGNOSTIC AUDIT INTO UI
============================================================ */

function renderStoredDiagnosticAudit() {

  const audit =
    loadDiagnosticAudit();


  if (
    audit &&
    Array.isArray(
      audit.records
    ) &&
    audit.records.length
  ) {

    renderDiagnosticAudit(
      audit
    );
  }
}
/* ============================================================
   EXPORT BENCHMARK JSON
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
    `AI-Trace-V67-Benchmark-${Date.now()}.json`;


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
   EXPORT BENCHMARK CSV
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

    'segmentRange',

    'tmrRawLabel',

    'tmrRawScore',

    'tmrVote',

    'e5RawLabel',

    'e5RawScore',

    'e5Vote',

    'modernRawLabel',

    'modernRawScore',

    'modernVote'

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

        record.scores
          ?.tmr,

        record.scores
          ?.e5,

        record.scores
          ?.modern,

        record.human
          ?.score,

        record.consensus
          ?.raw,

        record.consensus
          ?.calibrated,

        record.consensus
          ?.sufficiency
          ?.score,

        record.consensus
          ?.confidence,

        record.consensus
          ?.uncertainty,

        record.consensus
          ?.verdict,

        record.consensus
          ?.modelAgreement
          ?.agreement,

        record.consensus
          ?.modelSpread,

        record.consensus
          ?.segmentRange,

        record.diagnostics
          ?.tmr
          ?.dominant
          ?.label,

        record.diagnostics
          ?.tmr
          ?.dominant
          ?.score,

        record.diagnostics
          ?.tmr
          ?.vote,

        record.diagnostics
          ?.e5
          ?.dominant
          ?.label,

        record.diagnostics
          ?.e5
          ?.dominant
          ?.score,

        record.diagnostics
          ?.e5
          ?.vote,

        record.diagnostics
          ?.modern
          ?.dominant
          ?.label,

        record.diagnostics
          ?.modern
          ?.dominant
          ?.score,

        record.diagnostics
          ?.modern
          ?.vote

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
    `AI-Trace-V67-Benchmark-${Date.now()}.csv`;


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
   EXPORT DIAGNOSTIC AUDIT
============================================================ */

function exportDiagnosticJSON() {

  const audit =
    loadDiagnosticAudit();


  if (
    !audit ||
    !Array.isArray(
      audit.records
    ) ||
    !audit.records.length
  ) {

    alert(
      'Run Diagnostic Audit first.'
    );


    return;
  }


  const payload = {

    product:
      'AI Trace',

    version:
      VERSION,

    exportedAt:
      nowISO(),

    diagnosticAudit:
      audit

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
    `AI-Trace-V67-Diagnostics-${Date.now()}.json`;


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

    /*
      V6.7 keeps raw diagnostic information
      for recent scans as well.
    */

    diagnostics:
      scan.diagnostics,

    diagnosticWarnings:
      scan.diagnosticWarnings,

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
   OPEN DATASET IMPORTER
============================================================ */

function openDatasetImporter() {

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


/* ============================================================
   CORE EVENTS
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


      if (
        $('scanDiagnosticStatus')
      ) {

        $('scanDiagnosticStatus')
          .textContent =
          'WAITING';
      }


      if (
        $('scanRawDiagnostics')
      ) {

        $('scanRawDiagnostics')
          .innerHTML = `

<div class="ev">

  <small>
    Run a Smart Scan to inspect detector outputs.
  </small>

</div>

`;
      }
    }
  );


$('scan')
  ?.addEventListener(
    'click',
    runSmartScan
  );


/* ============================================================
   DATASET IMPORTER EVENTS
============================================================ */

$('openBulkImport')
  ?.addEventListener(
    'click',
    openDatasetImporter
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


$('chooseDatasetFile')
  ?.addEventListener(
    'click',
    () => {

      $('datasetFileInput')
        ?.click();
    }
  );


$('datasetFileInput')
  ?.addEventListener(
    'change',
    async event => {

      const file =
        event.target
          ?.files?.[
            0
          ];


      if (
        !file
      ) {

        return;
      }


      await handleDatasetFileSelection(
        file
      );
    }
  );


$('clearDatasetFile')
  ?.addEventListener(
    'click',
    clearSelectedDatasetFile
  );


$('bulkImportText')
  ?.addEventListener(
    'input',
    () => {

      const raw =
        $('bulkImportText')
          ?.value ||
        '';


      const parsed =
        parseDatasetContent(
          raw,
          selectedDatasetFile
            ?.name ||
          ''
        );


      renderImportPreview(
        parsed
      );
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


/* ============================================================
   CALIBRATION EVENTS
============================================================ */

$('runCalibrationQueue')
  ?.addEventListener(
    'click',
    runCalibrationWorker
  );


$('pauseCalibrationQueue')
  ?.addEventListener(
    'click',
    toggleCalibrationPause
  );


/* ============================================================
   V6.7 DIAGNOSTIC EVENTS
============================================================ */

$('runDiagnosticAudit')
  ?.addEventListener(
    'click',
    runDiagnosticAudit
  );


/* ============================================================
   EXPORT EVENTS
============================================================ */

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


$('exportDiagnosticJSON')
  ?.addEventListener(
    'click',
    exportDiagnosticJSON
  );


/* ============================================================
   CLEAR BENCHMARK
============================================================ */

$('clearBenchmark')
  ?.addEventListener(
    'click',
    () => {

      const confirmation =
        confirm(
          'Delete all AI Trace benchmark records from this browser?\n\nThis will also remove the V6.7 diagnostic audit.'
        );


      if (
        !confirmation
      ) {

        return;
      }


      localStorage.removeItem(
        BENCH_KEY
      );


      /*
        Calibration queue is deliberately kept.
        This prevents accidental destruction of imported
        but not yet analyzed samples.

        Diagnostic results are cleared because they no
        longer correspond to the dataset.
      */

      clearDiagnosticAudit();


      renderDatasetManager();


      renderCalibrationLab();


      if (
        $('diagnosticStatusBadge')
      ) {

        $('diagnosticStatusBadge')
          .textContent =
          'NOT AUDITED';
      }


      if (
        $('diagnosticSampleCount')
      ) {

        $('diagnosticSampleCount')
          .textContent =
          '0';
      }


      if (
        $('diagnosticMappingWarnings')
      ) {

        $('diagnosticMappingWarnings')
          .textContent =
          '0';
      }


      if (
        $('diagnosticModelFailures')
      ) {

        $('diagnosticModelFailures')
          .textContent =
          '0';
      }


      if (
        $('diagnosticPossibleInversions')
      ) {

        $('diagnosticPossibleInversions')
          .textContent =
          '0';
      }


      if (
        $('diagnosticDisagreements')
      ) {

        $('diagnosticDisagreements')
          .textContent =
          '0';
      }


      if (
        $('diagnosticCoverage')
      ) {

        $('diagnosticCoverage')
          .textContent =
          '0%';
      }


      if (
        $('labelMappingStatus')
      ) {

        $('labelMappingStatus')
          .innerHTML = `

<div class="ev">

  <div class="evTop">

    <span>
      Awaiting diagnostic audit
    </span>

    <span>
      UNKNOWN
    </span>

  </div>

  <small>
    Raw model labels have not yet been compared against known-origin benchmark samples.
  </small>

</div>

`;
      }


      if (
        $('detectorRawDiagnostics')
      ) {

        $('detectorRawDiagnostics')
          .innerHTML = `

<div class="ev">

  <small>
    Run Diagnostic Audit to inspect TMR, E5-small and ModernBERT raw outputs.
  </small>

</div>

`;
      }


      if (
        $('diagnosticAIControls')
      ) {

        $('diagnosticAIControls')
          .innerHTML = `

<div class="ev">

  <small>
    No diagnostic AI-control results yet.
  </small>

</div>

`;
      }


      if (
        $('diagnosticHumanControls')
      ) {

        $('diagnosticHumanControls')
          .innerHTML = `

<div class="ev">

  <small>
    No diagnostic HUMAN-control results yet.
  </small>

</div>

`;
      }


      if (
        $('diagnosticRecords')
      ) {

        $('diagnosticRecords')
          .innerHTML = `

<div class="ev">

  <small>
    No diagnostic records available.
  </small>

</div>

`;
      }


      if (
        $('diagnosticConclusion')
      ) {

        $('diagnosticConclusion')
          .innerHTML = `

<div class="ev">

  <div class="evTop">

    <span>
      Diagnosis pending
    </span>

    <span>
      WAITING
    </span>

  </div>

  <small>
    AI Trace will not alter detector label mapping automatically until diagnostic evidence is available.
  </small>

</div>

`;
      }


      alert(
        'Benchmark dataset and diagnostic audit deleted.'
      );
    }
  );


/* ============================================================
   V6.7 DEVELOPER API
============================================================ */

window.AITraceV67 = {

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

      diagnostic:
        loadDiagnosticAudit(),

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


  queue() {

    return loadCalibrationQueue();
  },


  history() {

    return loadJSON(
      HISTORY_KEY,
      []
    );
  },


  diagnostic() {

    return loadDiagnosticAudit();
  },


  runDiagnostic() {

    return runDiagnosticAudit();
  },


  runQueue() {

    return runCalibrationWorker();
  },


  pauseQueue() {

    if (
      workerRunning
    ) {

      workerPaused =
        true;


      setWorkerUI(
        'Paused',
        'Calibration worker will pause before the next sample.'
      );
    }
  },


  resumeQueue() {

    if (
      workerRunning
    ) {

      workerPaused =
        false;


      setWorkerUI(
        'Running',
        'Calibration worker resumed.'
      );
    }
  },


  stopQueue() {

    workerStopRequested =
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
        item.status ===
        'FAILED'
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


  clearQueue() {

    const confirmation =
      confirm(
        'Delete the full calibration queue?'
      );


    if (
      !confirmation
    ) {

      return;
    }


    localStorage.removeItem(
      QUEUE_KEY
    );


    renderDatasetManager();
  },


  clearHistory() {

    localStorage.removeItem(
      HISTORY_KEY
    );
  },


  clearDiagnostics() {

    clearDiagnosticAudit();


    location.reload();
  },


  exportJSON() {

    exportBenchmarkJSON();
  },


  exportCSV() {

    exportBenchmarkCSV();
  },


  exportDiagnostics() {

    exportDiagnosticJSON();
  }

};


/* ============================================================
   BACKWARD COMPATIBILITY
============================================================ */

/*
  Keep a V6.6 alias temporarily so any old console commands
  or debugging shortcuts do not immediately break.
*/

window.AITraceV66 =
  window.AITraceV67;


/* ============================================================
   INITIALIZATION
============================================================ */

recoverInterruptedQueue();


updateCount();


renderDatasetFileInfo();


renderImportPreview({

  samples:
    [],

  errors:
    []

});


renderDatasetManager();


renderCalibrationLab();


renderStoredDiagnosticAudit();


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
      : initialQueue.failed
        ? 'Complete with errors'
        : 'Complete',

    `${initialQueue.pending} pending · ${initialQueue.complete} complete · ${initialQueue.failed} failed`,

    initialQueue.total
      ? processed /
        initialQueue.total *
        100
      : 0

  );
}


/* ============================================================
   STARTUP STATUS
============================================================ */

if (
  $('modelState')
) {

  $('modelState')
    .textContent =
    'V6.7 diagnostic engine standby';
}


console.info(
  `AI TRACE V${VERSION} loaded successfully`
);
