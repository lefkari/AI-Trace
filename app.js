import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL = 'onnx-community/tmr-ai-text-detector-ONNX';
let classifier = null;

const $ = id => document.getElementById(id);
const text = $('text');

text.oninput = () => {
  $('count').textContent =
    (text.value.trim() ? text.value.trim().split(/\s+/).length : 0) +
    ' words';
};

$('clear').onclick = () => {
  text.value = '';
  text.oninput();
  $('report').classList.add('hidden');
};

$('demo').onclick = () => {
  text.value = `Artificial intelligence is increasingly becoming an integral part of modern organizations. Moreover, its ability to process large volumes of information enables companies to identify patterns, improve decision-making, and automate repetitive processes. However, successful adoption requires more than simply deploying new technology. Organizations must also establish appropriate governance frameworks, train employees, monitor outcomes, and ensure that automated systems remain transparent and accountable. Ultimately, businesses that combine technological innovation with responsible implementation are more likely to create sustainable long-term value while minimizing operational and ethical risks.`;
  text.oninput();
};

$('scan').onclick = run;

function progress(n, t) {
  $('progress').classList.remove('hidden');
  $('bar').style.width = n + '%';
  $('progressText').textContent = t;
}

function language(t) {
  const latin = (t.match(/[A-Za-z]/g) || []).length;
  const total = (t.match(/\p{L}/gu) || []).length;

  return total && latin / total > 0.8
    ? 'English'
    : 'Non-English';
}

function profile(t) {
  const words = t.trim().split(/\s+/);

  const sentences = t
    .split(/[.!?]+/)
    .map(x => x.trim())
    .filter(Boolean);

  const clean = words
    .map(w =>
      w
        .toLowerCase()
        .replace(/[^\p{L}\p{N}]/gu, '')
    )
    .filter(Boolean);

  const avg =
    words.length /
    Math.max(1, sentences.length);

  const lex =
    new Set(clean).size /
    Math.max(clean.length, 1);

  const lens = sentences.map(
    s => s.split(/\s+/).length
  );

  const mean =
    lens.reduce((a, b) => a + b, 0) /
    Math.max(1, lens.length);

  const variance =
    lens.reduce(
      (a, b) => a + (b - mean) ** 2,
      0
    ) / Math.max(1, lens.length);

  const transitions =
    (
      t.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion)\b/gi
      ) || []
    ).length;

  return {
    words: words.length,
    sentences: sentences.length,
    avg,
    lex,
    variance,
    transitions
  };
}

function chunks(t, max = 1450) {
  const sentences =
    t.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ||
    [t];

  const out = [];
  let current = '';

  for (const sentence of sentences) {
    if (
      (current + sentence).length > max &&
      current
    ) {
      out.push(current.trim());
      current = sentence;
    } else {
      current += sentence;
    }
  }

  if (current.trim()) {
    out.push(current.trim());
  }

  return out.slice(0, 8);
}

async function model() {
  if (classifier) {
    return classifier;
  }

  $('modelState').textContent =
    'Loading local ML engine…';

  progress(
    18,
    'Downloading model on first use…'
  );

  classifier = await pipeline(
    'text-classification',
    MODEL,
    {
      dtype: 'q8'
    }
  );

  $('modelState').textContent =
    'TMR engine ready ✓';

  return classifier;
}

function probability(output) {
  const results =
    (
      Array.isArray(output)
        ? output
        : [output]
    ).flat();

  const ai = results.find(x =>
    /AI|LABEL_1|generated/i.test(
      x.label
    )
  );

  const human = results.find(x =>
    /human|LABEL_0/i.test(
      x.label
    )
  );

  if (ai) {
    return Number(ai.score);
  }

  if (human) {
    return 1 - Number(human.score);
  }

  return 0.5;
}

async function run() {
  const t = text.value.trim();

  const wordCount = t
    ? t.split(/\s+/).length
    : 0;

  if (wordCount < 80) {
    alert(
      'For a more meaningful result, paste at least 80 words.'
    );
    return;
  }

  $('scan').disabled = true;

  progress(
    5,
    'Building document profile…'
  );

  const p = profile(t);
  const lang = language(t);

  const scores = [];
  const parts = chunks(t);

  let ml = true;

  try {
    const classifierModel =
      await model();

    for (
      let i = 0;
      i < parts.length;
      i++
    ) {
      progress(
        35 +
          Math.round(
            (i / parts.length) * 45
          ),
        `Analyzing segment ${
          i + 1
        }/${parts.length}…`
      );

      const output =
        await classifierModel(
          parts[i],
          {
            top_k: null,
            truncation: true
          }
        );

      scores.push(
        Math.round(
          probability(output) * 100
        )
      );
    }
  } catch (error) {
    console.error(error);

    ml = false;

    $('modelState').textContent =
      'ML unavailable • evidence fallback';

    for (let i = 0; i < parts.length; i++) {
      scores.push(
        heuristic(p)
      );
    }
  }

  const raw =
    Math.round(
      scores.reduce(
        (a, b) => a + b,
        0
      ) / scores.length
    );

  const uncertainty =
    calcUncertainty(
      scores,
      p,
      lang,
      ml
    );

  const confidence =
    Math.max(
      20,
      100 - uncertainty
    );

  render(
    raw,
    confidence,
    uncertainty,
    p,
    parts,
    scores,
    lang,
    ml
  );

  progress(
    100,
    'Trace complete'
  );

  setTimeout(() => {
    $('progress').classList.add(
      'hidden'
    );
  }, 400);

  $('scan').disabled = false;
}

function heuristic(p) {
  let score = 42;

  if (p.variance < 35) {
    score += 10;
  }

  if (p.transitions > 2) {
    score += 8;
  }

  if (
    p.avg > 18 &&
    p.avg < 32
  ) {
    score += 8;
  }

  return Math.min(
    78,
    score
  );
}

function calcUncertainty(
  scores,
  p,
  lang,
  ml
) {
  let uncertainty = 18;

  if (!ml) {
    uncertainty += 35;
  }

  if (lang !== 'English') {
    uncertainty += 28;
  }

  if (p.words < 150) {
    uncertainty += 15;
  }

  const mean =
    scores.reduce(
      (a, b) => a + b,
      0
    ) / scores.length;

  const spread =
    Math.sqrt(
      scores.reduce(
        (a, b) =>
          a + (b - mean) ** 2,
        0
      ) / scores.length
    );

  uncertainty += Math.min(
    20,
    spread
  );

  return Math.min(
    90,
    Math.round(uncertainty)
  );
}

function escapeHTML(s) {
  return s.replace(
    /[&<>"']/g,
    c =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      }[c])
  );
}

function render(
  score,
  confidence,
  uncertainty,
  p,
  parts,
  scores,
  lang,
  ml
) {
  $('report').classList.remove(
    'hidden'
  );

  $('score').textContent =
    score + '%';

  $('scaleFill').style.width =
    score + '%';

  $('verdict').textContent =
    score >= 75
      ? 'Strong AI signal'
      : score >= 55
      ? 'Moderate AI signal'
      : score >= 35
      ? 'Mixed / uncertain'
      : 'Low AI signal';

  $('confidence').textContent =
    `Confidence: ${
      confidence >= 70
        ? 'High'
        : confidence >= 45
        ? 'Medium'
        : 'Low'
    } (${confidence}%)`;

  $('explain').textContent =
    lang === 'English' && ml
      ? 'Score is driven primarily by the RAID-trained TMR classifier, with uncertainty adjusted for document length and segment disagreement.'
      : 'This content is outside the strongest validated path; interpret the result cautiously.';

  const human =
    100 - score;

  $('humanVal').textContent =
    human + '%';

  $('aiVal').textContent =
    score + '%';

  $('uncertainVal').textContent =
    uncertainty + '%';

  $('humanBar').style.width =
    human + '%';

  $('aiBar').style.width =
    score + '%';

  $('uncertainBar').style.width =
    uncertainty + '%';

  $('engineBadge').textContent =
    ml
      ? 'TMR • ONNX'
      : 'FALLBACK';

  const evidence = [
    [
      'ML classifier',
      ml
        ? `${score}% aggregate machine-generated-text signal across ${scores.length} segment(s).`
        : 'Model unavailable; result uses fallback signals.',
      ml
        ? 'Primary'
        : 'Fallback'
    ],
    [
      'Segment agreement',
      `${
        Math.max(...scores) -
        Math.min(...scores)
      } point range between analyzed chunks. Lower spread generally increases confidence.`,
      'Evidence'
    ],
    [
      'Language fit',
      lang === 'English'
        ? 'English detected — matches the detector’s primary training language.'
        : 'Non-English detected — detector reliability is reduced.',
      'Context'
    ],
    [
      'Structural consistency',
      p.variance < 35
        ? 'Sentence lengths are relatively consistent.'
        : 'Sentence lengths show substantial variation.',
      'Supporting'
    ]
  ];

  $('evidence').innerHTML =
    evidence
      .map(
        item => `
        <div class="ev">
          <div class="evTop">
            <span>${item[0]}</span>
            <span>${item[2]}</span>
          </div>
          <small>${item[1]}</small>
        </div>
      `
      )
      .join('');

  const metrics = {
    Words: p.words,
    Sentences: p.sentences,
    'Avg. words / sentence':
      p.avg.toFixed(1),
    'Lexical diversity':
      Math.round(
        p.lex * 100
      ) + '%',
    Language: lang,
    'Segments analyzed':
      scores.length
  };

  $('metrics').innerHTML =
    Object.entries(metrics)
      .map(
        ([key, value]) => `
        <div class="metric">
          <span>${key}</span>
          <b>${value}</b>
        </div>
      `
      )
      .join('');

  $('segments').innerHTML =
    parts
      .map(
        (part, i) => `
        <div class="segment">
          <div class="segmentHead">
            <b>Segment ${i + 1}</b>
            <span>${scores[i]}% AI signal</span>
          </div>

          <div class="segmentMeter">
            <i style="width:${scores[i]}%"></i>
          </div>

          <p>
            ${escapeHTML(
              part.slice(0, 300)
            )}
            ${
              part.length > 300
                ? '…'
                : ''
            }
          </p>
        </div>
      `
      )
      .join('');

  $('report').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}
