// Builds slides/talk_simglucose.pptx: a ~10-minute talk on the Simglucose results.
// Look follows the Keynote "Basic White" template we were given: white slides, bold
// Helvetica Neue titles top-left, a bold subtitle line, few words, large figures.
// Every number comes from RESULTS.md / results/*.md (see the speaker notes for the CIs).
//
//   python slides/make_figures.py
//   node slides/build_deck.js        (npm install pptxgenjs, if it is missing)
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const IMG = (f) => path.join(HERE, "img", f);
const OUT = path.join(HERE, "talk_simglucose.pptx");

// Palette: the study's arm colours (analysis/common.py) on the template's black and white.
const INK = "000000", GRAY = "6E6E73", LIGHT = "F2F2F2", WHITE = "FFFFFF";
const BLUE = "2A78D6", ORANGE = "EB6834";                 // global σ, state-dependent σ(s)
const BLUE_T = "E4EEFA", ORANGE_T = "FCE7DE";               // tints for boxes
const RED = "B03A2E", RED_L = "EC9A90", GREEN = "4E9F5D", AMBER = "EDB451";
const FONT = "Helvetica Neue";

const THEME = {
  name: "Basic White (sigma study)", headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: INK, lt1: WHITE, dk2: "1D1D1F", lt2: LIGHT, accent1: BLUE, accent2: ORANGE, accent3: GREEN,
            accent4: RED, accent5: GRAY, accent6: AMBER, hlink: BLUE, folHlink: "8A5CD6" },
};

// pixel sizes of the images (for aspect ratios)
const PX = {
  "noise_global.png": [920, 520], "noise_state.png": [920, 520], "curves_shared.png": [2400, 980],
  "curves_perarm.png": [2400, 980], "min_logsigma.png": [690, 440], "sigma_vs_cgm.png": [792, 414],
  "sigma_episodes.png": [975, 510], "executed_noise_vs_cgm.png": [792, 419],
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, 16:9 like the template
pres.author = "Marina Lin, Rosa Wu";
pres.title = "Should exploration noise depend on the state?";
pres.subject = "Global vs. state-dependent sigma in PPO, Simglucose results";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

const L = 0.7, W = 13.333 - 2 * L; // left margin, content width

// ---- layouts ---------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: L, y: 1.45, w: W, h: 2.25, fontSize: 50, bold: true,
                                color: INK, align: "left", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: L, y: 3.85, w: W, h: 1.0, fontSize: 26, bold: true,
                                color: INK, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: L, y: 0.42, w: W, h: 0.75, fontSize: 40, bold: true,
                                color: INK, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: L, y: 1.17, w: W, h: 0.5, fontSize: 24, bold: true,
                                color: INK, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "STATEMENT", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 1.7, w: 13.333 - 1.8, h: 2.3, fontSize: 46,
                                bold: true, color: INK, align: "center", valign: "middle", margin: 0 }, text: "" } },
  ],
});

// ---- helpers ---------------------------------------------------------------------------
function content(title, subtitle) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  if (subtitle) s.addText(subtitle, { placeholder: "subtitle" });
  return s;
}
function text(s, t, o) {
  s.addText(t, Object.assign({ isTextBox: true, margin: 0, color: INK, fontSize: 24, valign: "top" }, o));
}
function image(s, file, x, y, w, altText) {
  const [pw, ph] = PX[file];
  const h = (w * ph) / pw;
  s.addImage({ path: IMG(file), x, y, w, h, altText });
  return h;
}
const run = (t, o = {}) => ({ text: t, options: o });

// ---- 1. title ---------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "TITLE" });
  s.addText("Should exploration noise depend on the state?", { placeholder: "title" });
  s.addText("Global vs. state-dependent σ in PPO, tested on insulin dosing for type 1 diabetes",
            { placeholder: "subtitle" });
  text(s, "Marina Lin  ·  Rosa Wu", { x: L, y: 5.55, w: 8, h: 0.4, fontSize: 22, bold: true });
  text(s, "Harvard College", { x: L, y: 5.97, w: 8, h: 0.35, fontSize: 18, bold: true });
  text(s, "November 2026", { x: 4.17, y: 6.75, w: 5, h: 0.35, fontSize: 16, bold: true, align: "center" });
  s.addNotes(
    "Hi! Our project asks a simple question about PPO, the most widely used policy-gradient method: " +
    "should the amount of exploration noise be the same in every state, or should the policy learn " +
    "to make it depend on the state? We tested this on three tasks; this talk focuses on the most " +
    "interesting one, controlling insulin for a simulated patient with type 1 diabetes.");
}

// ---- 2. the two parameterisations --------------------------------------------------------
{
  const s = content("PPO explores by adding Gaussian noise",
                    "Each action is sampled from N(μ(s), σ²). How wide should σ be?");
  const cw = 5.6, x2 = 13.333 - L - cw;
  text(s, "Global σ", { x: L, y: 2.05, w: cw, h: 0.55, fontSize: 28, bold: true });
  text(s, "One learned width for every state (the default in CleanRL and SB3)",
       { x: L, y: 2.62, w: cw, h: 0.85, fontSize: 22 });
  image(s, "noise_global.png", L, 3.6, 5.3, "Three bell curves of equal width: global sigma uses the same noise in every state");
  text(s, "State-dependent σ(s)", { x: x2, y: 2.05, w: cw, h: 0.55, fontSize: 28, bold: true });
  text(s, "The network outputs a width for each state", { x: x2, y: 2.62, w: cw, h: 0.85, fontSize: 22 });
  image(s, "noise_state.png", x2, 3.6, 5.3, "Three bell curves of different widths: sigma(s) can use a different noise level in each state");
  text(s, "Each curve is the action distribution in one state", { x: L, y: 6.72, w: W, h: 0.4, fontSize: 18,
       color: GRAY });
  s.addNotes(
    "PPO picks an action by sampling from a Gaussian: the mean mu(s) comes from the policy network, and " +
    "sigma sets how much it explores. Almost every implementation (CleanRL, Stable-Baselines3, OpenAI " +
    "Baselines) learns sigma as one global number per action dimension: the same width in every state (left). " +
    "The alternative is to let the network output sigma(s), so the width can change with the state (right). " +
    "That costs one extra output layer.");
}

// ---- 3. the bet --------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "STATEMENT" });
  s.addText("Explore where mistakes are cheap.\nBe precise where they are costly.", { placeholder: "title" });
  text(s, "Andrychowicz et al. (2021) found no benefit from σ(s) on MuJoCo locomotion. We test tasks " +
          "where the precision you need clearly changes from state to state.",
       { x: 1.6, y: 4.55, w: 13.333 - 3.2, h: 1.3, fontSize: 22, color: GRAY, align: "center" });
  s.addNotes(
    "The intuition for sigma(s) is this: explore a lot where mistakes are cheap, and act precisely where " +
    "they are costly. A large study by Andrychowicz et al. (2021, 'What matters in on-policy RL') found no " +
    "benefit from state-dependent sigma on MuJoCo locomotion. But locomotion may not need different " +
    "precision in different states. So we picked tasks that clearly do.");
}

// ---- 4. the task -------------------------------------------------------------------------
{
  const s = content("Insulin dosing: a precision gradient",
                    "Simglucose: a simulated adolescent with type 1 diabetes");
  text(s, [
    run("Every 3 minutes, the agent sets the insulin rate (0–10× normal basal)", { bullet: true, breakLine: true }),
    run("It sees the last hour of glucose and insulin; meals come at random times", { bullet: true, breakLine: true }),
    run("One episode = one simulated day (480 decisions)", { bullet: true }),
  ], { x: L, y: 1.98, w: W, h: 1.75, fontSize: 24, paraSpaceAfter: 8 });

  // glucose zones (schematic, not to scale)
  const y = 5.0, h = 0.72;
  const zones = [
    { x: L, w: 1.9, fill: RED, label: "< 54", name: "severe low", tc: WHITE },
    { x: L + 1.9, w: 1.6, fill: RED_L, label: "54–70", name: "low", tc: INK },
    { x: L + 3.5, w: 4.8, fill: GREEN, label: "70–180 mg/dL", name: "target range", tc: WHITE },
    { x: L + 8.3, w: W - 8.3, fill: AMBER, label: "> 180", name: "high", tc: INK },
  ];
  for (const z of zones) {
    s.addShape(pres.shapes.RECTANGLE, { x: z.x, y, w: z.w, h, fill: { color: z.fill }, line: { color: WHITE, width: 1.5 } });
    text(s, z.label, { x: z.x, y, w: z.w, h, fontSize: 20, color: z.tc, align: "center", valign: "middle" });
    text(s, z.name, { x: z.x, y: y + h + 0.08, w: z.w, h: 0.4, fontSize: 20, align: "center" });
  }
  text(s, "Extra insulin here: dangerous low", { x: L, y: 4.05, w: 5.6, h: 0.45, fontSize: 22 });
  s.addShape(pres.shapes.LINE, { x: L + 1.75, y: 4.52, w: 0, h: 0.42, line: { color: GRAY, width: 2.5, endArrowType: "triangle" } });
  text(s, "Extra insulin here: back toward range", { x: 5.8, y: 4.05, w: 13.333 - L - 5.8, h: 0.45, fontSize: 22,
       align: "right" });
  s.addShape(pres.shapes.LINE, { x: L + 8.3 + (W - 8.3) / 2, y: 4.52, w: 0, h: 0.42,
                                 line: { color: GRAY, width: 2.5, endArrowType: "triangle" } });
  text(s, "Clinical targets for a day: > 70% of the time in 70–180 mg/dL and < 4% below 70 (international consensus)",
       { x: L, y: 6.55, w: W, h: 0.45, fontSize: 18, color: GRAY });
  s.addNotes(
    "Our deep-dive task is Simglucose, a Python version of the FDA-accepted UVA/Padova type 1 diabetes " +
    "simulator. Every 3 simulated minutes the agent chooses a basal insulin rate, from zero up to ten times " +
    "the patient's normal rate. It observes the last hour of glucose and insulin plus the time of day, and meals arrive at random. " +
    "An episode is one day. The key property: the same extra insulin is dangerous when glucose is already low " +
    "(it can cause hypoglycemia), while at high glucose it just brings glucose back toward the target range. So if any task should reward a " +
    "state-dependent sigma, it is this one. We count a day as a success if it meets the clinical consensus " +
    "targets: more than 70% of the time in range and less than 4% below 70 mg/dL.");
}

// ---- 5. setup ----------------------------------------------------------------------------
{
  const s = content("Same agent, one difference",
                    "PPO (CleanRL): same network, init, seeds and hyperparameters");
  const by = 2.15, bh = 1.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: L, y: by, w: 2.75, h: bh, rectRadius: 0.12, fill: { color: LIGHT },
                                             line: { color: LIGHT } });
  text(s, [run("state", { bold: true, fontSize: 24, breakLine: true }), run("last hour of glucose and insulin", { fontSize: 18 })],
       { x: L + 0.15, y: by, w: 2.45, h: bh, align: "center", valign: "middle" });
  s.addShape(pres.shapes.LINE, { x: L + 2.8, y: by + bh / 2, w: 0.6, h: 0, line: { color: GRAY, width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.15, y: by, w: 2.75, h: bh, rectRadius: 0.12, fill: { color: LIGHT },
                                             line: { color: LIGHT } });
  text(s, [run("policy network", { bold: true, fontSize: 24, breakLine: true }), run("2 layers × 64, tanh", { fontSize: 18 })],
       { x: 4.3, y: by, w: 2.45, h: bh, align: "center", valign: "middle" });
  // outputs
  const ox = 7.75, ow = 13.333 - L - ox;
  // arrows: up to the mean, down to the pair of σ variants (shapes never get a negative size)
  s.addShape(pres.shapes.LINE, { x: 6.95, y: 2.3, w: 0.75, h: 0.4, flipV: true, line: { color: GRAY, width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: 6.95, y: 3.05, w: 0.75, h: 0.55, line: { color: GRAY, width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ox, y: 1.95, w: ow, h: 0.62, rectRadius: 0.1, fill: { color: LIGHT }, line: { color: LIGHT } });
  text(s, [run("mean dose μ(s)", { bold: true }), run("   same in both", { color: GRAY, fontSize: 20 })],
       { x: ox + 0.2, y: 1.95, w: ow - 0.3, h: 0.62, fontSize: 22, valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ox, y: 2.8, w: ow, h: 0.6, rectRadius: 0.1, fill: { color: BLUE_T }, line: { color: BLUE, width: 1.5 } });
  text(s, [run("global σ: ", { bold: true, color: BLUE }), run("one learned number")],
       { x: ox + 0.2, y: 2.8, w: ow - 0.3, h: 0.6, fontSize: 22, valign: "middle" });
  text(s, "or", { x: ox + 0.2, y: 3.42, w: 1.0, h: 0.34, fontSize: 18, color: GRAY, valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ox, y: 3.78, w: ow, h: 0.6, rectRadius: 0.1, fill: { color: ORANGE_T }, line: { color: ORANGE, width: 1.5 } });
  text(s, [run("σ(s): ", { bold: true, color: ORANGE }), run("an extra output layer")],
       { x: ox + 0.2, y: 3.78, w: ow - 0.3, h: 0.6, fontSize: 22, valign: "middle" });

  text(s, [
    run("10 seeds per arm, 1M steps per run (about 5.7 simulated years)", { bullet: true, breakLine: true }),
    run("Tuned with Optuna: 40 trials, both arms trained on each", { bullet: true, breakLine: true }),
    run("Reported: IQM over seeds with 95% bootstrap CIs", { bullet: true }),
  ], { x: L, y: 4.95, w: W, h: 1.85, fontSize: 24, paraSpaceAfter: 8 });
  s.addNotes(
    "The two arms are the same agent: PPO from CleanRL, the same networks, the same initialization (at the " +
    "start both have exactly the same sigma in every state), the same clamp on log sigma, and the same " +
    "hyperparameters. The only difference is where sigma comes from. We ran 10 seeds per arm, one million " +
    "steps each, which is about 5.7 years of simulated life. Hyperparameters were tuned with Optuna: 40 " +
    "trials, and every trial trained both arms, so the comparison is paired. We report the interquartile mean " +
    "over seeds with 95% bootstrap confidence intervals.");
}

// ---- 6. headline numbers -----------------------------------------------------------------
{
  const s = content("Same hyperparameters: global σ is safer",
                    "Best shared config · IQM over 10 seeds · 60 test days per seed");
  const hdr = (lines) => ({ text: lines.map((t, i) => run(t, i < lines.length - 1 ? { breakLine: true } : {})),
                            options: { bold: true, align: "center", fill: { color: LIGHT }, fontSize: 22 } });
  const num = (t) => ({ text: t, options: { align: "center" } });
  s.addTable([
    [{ text: "", options: { fill: { color: LIGHT } } }, hdr(["Days meeting the", "clinical targets"]),
     hdr(["Days with a severe low", "(glucose < 54\u00A0mg/dL)"])],
    [{ text: "Global σ" }, num("96%"), num("0%")],
    [{ text: "State-dependent σ(s)" }, num("61%"), num("14%")],
  ], { x: L, y: 2.25, w: W, colW: [3.93, 4.0, 4.0], rowH: [1.1, 0.95, 0.95], fontSize: 26, color: INK,
       fontFace: FONT, valign: "middle", margin: [0, 0.15, 0, 0.15], border: { type: "solid", pt: 1, color: "D9D9D9" } });
  text(s, "σ(s) − global: −35 points [−61, −12] and +14 points [+2, +39], 95% bootstrap CIs",
       { x: L, y: 5.75, w: W, h: 0.45, fontSize: 18, color: GRAY });
  s.addNotes(
    "Here is the headline. With the best shared hyperparameters, so the arms differ only in sigma, global sigma " +
    "met the clinical targets on 96% of evaluation days, the state-dependent policy on only 61%. The " +
    "difference is -35 percentage points, 95% CI [-61, -12]. Severe lows, glucose below 54 mg/dL at least " +
    "once in the day, happened on 0% of days with global sigma and 14% with sigma(s), CI [+2, +39]. In return " +
    "(1 minus the Magni risk index per step), global sigma is also ahead: 474.8 vs 465.4, difference -9.3 " +
    "[-34.2, -3.2].");
}
// ---- 7. learning curves (shared) ---------------------------------------------------------
{
  const s = content("σ(s) learns slower and less reliably",
                    "Bold: IQM over 10 seeds · band: 95% CI · thin lines: single seeds");
  image(s, "curves_shared.png", 0.67, 1.95, 12.0, "Learning curves at the shared config: global sigma reaches about 96% of days on target and no severe lows; sigma(s) ends near 61% with severe lows on about 14% of days");
  s.addNotes(
    "Over training, global sigma (blue) reaches the targets on almost every day after about 300k steps and " +
    "stays there; every seed gets there. Sigma(s) (orange) climbs slowly, and the seeds spread out widely: " +
    "some never get going. Panel b shows the safety side: severe lows disappear quickly for global sigma, " +
    "while sigma(s) still has them at the end of training.");
}

// ---- 8. optimization difficulty ----------------------------------------------------------
{
  const s = content("σ(s) is harder to optimize", "Training diagnostics at the same shared config");
  const h = image(s, "min_logsigma.png", L, 2.05, 6.5, "Smallest log sigma across states during training: about -0.6 for global sigma, falling to the floor of -5 for sigma(s)");
  text(s, [run("Smallest log σ across states, during training", { breakLine: true }),
           run("Blue: global σ", { color: BLUE, bold: true }), run("     "), run("Orange: σ(s)", { color: ORANGE, bold: true })],
       { x: L, y: 2.05 + h + 0.1, w: 6.6, h: 0.75, fontSize: 18, color: GRAY });
  const x = 7.75, w = 13.333 - L - x;
  text(s, [
    run("In all 10 σ(s) runs, σ hit its floor (e⁻⁵) in some states. Global\u00A0σ:\u00A0never", { bullet: true, breakLine: true }),
    run("25% of σ(s) updates had a KL spike above 0.02. Global\u00A0σ:\u00A0none", { bullet: true, breakLine: true }),
    run("3 of 10 σ(s) runs never reached 50% of days on target. Global\u00A0σ:\u00A00", { bullet: true }),
  ], { x, y: 2.1, w, h: 4.6, fontSize: 22, paraSpaceAfter: 16 });
  s.addNotes(
    "Why does sigma(s) do worse? It is harder to optimize. Left: in every sigma(s) run, sigma in some states " +
    "is driven all the way down to its floor (log sigma = -5), while global sigma settles around 0.55. A " +
    "nearly deterministic action makes PPO's probability ratios extreme: 25% of sigma(s) updates had a KL " +
    "spike above 0.02, versus none for global sigma. 3 of 10 sigma(s) runs never met the targets on even half " +
    "the evaluation days, and the spread of final return across seeds is 26.5 vs 0.7.");
}

// ---- 9. per-arm tuning -------------------------------------------------------------------
{
  const s = content("With its own tuning, σ(s) catches up",
                    "Each arm at its own best config: 99% vs 96% of days on target");
  image(s, "curves_perarm.png", 1.17, 1.95, 11.0, "Learning curves with each arm at its own best config: both reach about 96-99% of days on target and no severe lows; sigma(s) is slower early on");
  text(s, "No severe lows for either arm. σ(s) wanted a 5× higher learning rate and 2× longer rollouts",
       { x: L, y: 6.6, w: W, h: 0.45, fontSize: 20, color: GRAY });
  s.addNotes(
    "But that is not the end of the story. When each arm gets its own best hyperparameters, sigma(s) catches " +
    "up: 99% of days on target versus 96% for global sigma, difference +2.5 points with CI [-0.6, +4.4], so no " +
    "clear difference, and neither arm has severe lows. Sigma(s) wanted a 5x higher learning rate (2.7e-3 vs " +
    "5.3e-4) and 2x longer rollouts (512 vs 256 steps per environment). It still learns more slowly early on. " +
    "In return, global sigma stays ahead by only 0.16% (475.1 vs 474.4, CI [-1.4, -0.2]).");
}

// ---- 10. tuning vs parameterisation -------------------------------------------------------
{
  const s = content("Tuning mattered more than the σ choice", "Days meeting the clinical targets, IQM over 10 seeds");
  s.addChart(pres.charts.BAR, [
    { name: "Global σ", labels: ["Untuned", "Tuned (shared)", "Tuned (own)"], values: [0.039, 0.958, 0.964] },
    { name: "State-dependent σ(s)", labels: ["Untuned", "Tuned (shared)", "Tuned (own)"], values: [0.0, 0.611, 0.989] },
  ], {
    x: 0.45, y: 1.85, w: 8.6, h: 5.15, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
    chartColors: [BLUE, ORANGE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%",
    dataLabelFontSize: 20, dataLabelFontFace: "+mn-lt", dataLabelColor: INK,
    catAxisLabelFontSize: 20, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: INK, catAxisLineShow: true,
    catAxisLineColor: "BBBBBB", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.1,
    valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 20, legendFontFace: "+mn-lt", legendColor: INK,
  });
  const x = 9.45, w = 13.333 - L - x;
  text(s, "Across 26 paired tuning trials, σ(s) scored higher in half of them", { x, y: 2.4, w, h: 1.6, fontSize: 22 });
  text(s, "Wilcoxon p\u00A0=\u00A00.62: no systematic winner", { x, y: 4.1, w, h: 0.9, fontSize: 20, color: GRAY });
  s.addNotes(
    "Putting the three settings side by side: untuned, with the spec's default hyperparameters, neither arm " +
    "learns to dose safely (4% vs 0% of days on target). With the shared tuned config, global sigma is far " +
    "ahead. With each arm's own config, the two are equal. And across the 26 completed tuning trials, where " +
    "both arms were trained with the same hyperparameters, sigma(s) scored higher in exactly half (Wilcoxon " +
    "p = 0.62). So the hyperparameters matter much more than how sigma is parameterized.");
}

// ---- 11. sigma vs glucose ----------------------------------------------------------------
{
  const s = content("σ(s) is noisiest when glucose is low",
                    "We expected the opposite: the smallest σ at low glucose");
  const h = image(s, "sigma_vs_cgm.png", L, 2.0, 8.0, "Mean sigma versus glucose: flat at about 0.55 for global sigma; for sigma(s) about 2 at low glucose, about 1.1 in range");
  text(s, "σ of the insulin action vs. glucose (CGM), shared config; bold: mean over 10 seeds",
       { x: L, y: 2.0 + h + 0.1, w: 8.0, h: 0.45, fontSize: 18, color: GRAY });
  const x = 9.2, w = 13.333 - L - x;
  text(s, [
    run("At 55–80 mg/dL: σ(s)\u00A0≈\u00A02.1", { bullet: true, breakLine: true }),
    run("At 140–200 mg/dL: σ(s)\u00A0≈\u00A01.1", { bullet: true, breakLine: true }),
    run("Global\u00A0σ\u00A0≈\u00A00.55 everywhere", { bullet: true }),
  ], { x, y: 2.1, w, h: 3.6, fontSize: 22, paraSpaceAfter: 16 });
  s.addNotes(
    "Now the second question: what does sigma(s) actually learn? It does become state-dependent: 97% of the " +
    "variance of log sigma is across states within a run. But not in the way we predicted. We expected small " +
    "sigma at low glucose, where an overdose is dangerous. Instead sigma(s) is largest there: about 2 at " +
    "55-80 mg/dL versus about 1.1 in range, and roughly twice its value in range overall. Global sigma is 0.55 " +
    "everywhere. The rank correlation between sigma and the cost of noise is only -0.04.");
}

// ---- 12. sigma over a day ----------------------------------------------------------------
{
  const s = content("Low glucose sends σ(s) to its ceiling",
                    "Two evaluation days of one σ(s) policy (training seed 1)");
  image(s, "sigma_episodes.png", L, 1.95, 9.0, "Sigma and glucose over two evaluation days: sigma(s) jumps to its ceiling once glucose falls below 100 mg/dL");
  const x = 10.05, w = 13.333 - L - x;
  text(s, [
    run("Top: σ of the insulin action", { breakLine: true }), run(" ", { fontSize: 10, breakLine: true }),
    run("Bottom: glucose (mg/dL)", { breakLine: true }), run(" ", { fontSize: 10, breakLine: true }),
    run("Dashed line: glucose falls below\u00A0100", { breakLine: true }), run(" ", { fontSize: 10, breakLine: true }),
    run("Ceiling: σ = e² ≈ 7.4"),
  ], { x, y: 2.2, w, h: 4.2, fontSize: 20 });
  s.addNotes(
    "Here is what that looks like within a day, for one sigma(s) policy. Top: sigma; bottom: glucose. While " +
    "glucose is above 100, sigma stays small. Once glucose falls below 100 (dashed line), sigma starts hitting " +
    "its ceiling of e^2, about 7.4, and flips between the ceiling and small values from one step to the next.");
}

// ---- 13. why ----------------------------------------------------------------------------
{
  const s = content("Why? Most of the noise is clipped away",
                    "What reaches the patient after clipping to the dose limits");
  const h = image(s, "executed_noise_vs_cgm.png", L, 2.0, 8.0, "Spread of the delivered action versus glucose: far below the policy sigma for both arms; at low glucose about 0.38 for sigma(s) and 0.22 for global sigma");
  text(s, "Solid: spread of the delivered dose · dotted: the policy's σ (action units, shared config)",
       { x: L, y: 2.0 + h + 0.1, w: 8.0, h: 0.75, fontSize: 18, color: GRAY });
  const x = 9.2, w = 13.333 - L - x;
  text(s, [
    run("The mean dose is outside the allowed range in 64% of states", { bullet: true, breakLine: true }),
    run("So a wide σ costs little, and σ(s) has little reason to shrink it", { bullet: true, breakLine: true }),
    run("At low glucose, σ(s) still lets 1.7× more insulin noise through", { bullet: true }),
  ], { x, y: 2.05, w, h: 4.6, fontSize: 22, paraSpaceAfter: 14 });
  s.addNotes(
    "Why would a learned sigma put the most noise exactly where it seems most dangerous? Because of the " +
    "action limits. In about 64% of states, for both arms, the mean action is outside the allowed range, so " +
    "it is clipped to zero insulin or to the maximum. When the mean is far outside the range, most samples are " +
    "clipped to the same dose, so even a very large sigma barely changes what the patient gets (solid lines vs " +
    "dotted). Noise there is cheap, so the gradient has little reason to shrink sigma. Our noise probe agrees: " +
    "for the global-sigma policies, noise at low glucose costs about 0.65x what it costs elsewhere. But clipping does not hide " +
    "everything: at low glucose sigma(s) still delivers about 1.7x more insulin noise than global sigma (0.38 vs " +
    "0.22 in action units, i.e. about 1.9x vs 1.1x the basal rate), which is consistent with its extra severe lows.");
}

// ---- 14. other tasks ----------------------------------------------------------------------
{
  const s = content("Three tasks, no consistent winner", "Shared tuned config, IQM over 10 seeds");
  const hdr = (t, al = "left") => ({ text: t, options: { bold: true, color: INK, fill: { color: LIGHT }, align: al } });
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ color: INK }, o) });
  s.addTable([
    [hdr("Task"), hdr("Metric"), hdr("Global σ", "center"), hdr("σ(s)", "center"), hdr("Verdict")],
    [cell("Pendulum swing-up"), cell("return"), cell("−185", { align: "center" }), cell("−176", { align: "center" }),
     cell("σ(s) barely ahead")],
    [cell("Parking a car"), cell("% parked"), cell("48%", { align: "center" }), cell("36%", { align: "center" }),
     cell("no clear difference")],
    [cell("Insulin dosing"), cell("% days on target"), cell("96%", { align: "center" }), cell("61%", { align: "center" }),
     cell("global σ ahead")],
  ], { x: L, y: 2.05, w: W, colW: [3.05, 2.85, 1.55, 1.45, 3.03], rowH: 0.72, fontSize: 22, fontFace: FONT,
       valign: "middle", margin: [0, 0.12, 0, 0.12], border: { type: "solid", pt: 1, color: "D9D9D9" } });
  text(s, "Untuned, σ(s) was far worse on Pendulum and neither arm learned to park. Tuned, the two arms end up " +
          "close in every task.", { x: L, y: 5.35, w: W, h: 1.0, fontSize: 22 });
  s.addNotes(
    "For context, the other two tasks. Pendulum: sigma(s) is barely ahead, +8.4 return with CI [+0.2, +18.6], " +
    "and that disappears at sigma(s)'s own best config. Parking: 48% vs 36% parked, CI [-47, +29], no clear " +
    "difference. Insulin is the only task with a clear gap, and as we saw, it closes with per-arm tuning. " +
    "Untuned, sigma(s) was much worse on Pendulum (-378 vs -193) and nothing parked. So across tasks, " +
    "the pattern is the same: no consistent winner once both are tuned.");
}

// ---- 15. takeaways -----------------------------------------------------------------------
{
  const s = content("Takeaways");
  const items = [
    ["Same hyperparameters: ", "global σ was safer (96%\u00A0vs\u00A061% of days on target, 0%\u00A0vs\u00A014% with severe lows)"],
    ["Own hyperparameters: ", "σ(s) catches up (99%\u00A0vs\u00A096%). Tuning mattered more than how σ is parameterized"],
    ["σ(s) varies with the state, but not as we hoped: ", "noisiest at low glucose, where clipping at the dose limits hides most of the noise"],
  ];
  items.forEach(([lead, rest], i) => {
    const y = 1.85 + i * 1.6;
    text(s, `${i + 1}.`, { x: L, y, w: 0.6, h: 0.5, fontSize: 26 });
    text(s, [run(lead, { bold: true }), run(rest)], { x: L + 0.6, y, w: W - 0.6, h: 1.35, fontSize: 26 });
  });
  s.addNotes(
    "Three takeaways. One: with the same hyperparameters, a global sigma was clearly safer for insulin dosing. " +
    "Two: a state-dependent sigma is harder to optimize, but with its own hyperparameters it catches up, so " +
    "tuning mattered more than the parameterization; this matches Andrychowicz et al. on locomotion. Three: " +
    "sigma(s) does learn to vary with the state, but not in the intuitive way. Because the dose is clipped at " +
    "its limits, most of the noise never reaches the patient where the policy is saturated, so noise looks cheap there, " +
    "and sigma(s) ends up noisiest at low glucose.");
}
// ---- 16. thanks --------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "TITLE" });
  s.addText("Thank you!", { placeholder: "title" });
  s.addText("Questions?", { placeholder: "subtitle" });
  text(s, [run("github.com/rosawoo/ppo_global_sd", { hyperlink: { url: "https://github.com/rosawoo/ppo_global_sd" },
                                                      color: BLUE })],
       { x: L, y: 5.55, w: 8, h: 0.45, fontSize: 22 });
  text(s, "Code, all results (RESULTS.md) and every figure", { x: L, y: 6.02, w: 8, h: 0.4, fontSize: 18, color: GRAY });
  s.addNotes("Thanks! Everything (code, results and figures) is in the repository.");
}

// ---- write -------------------------------------------------------------------------------
(async () => {
  await pres.writeFile({ fileName: OUT });
  // Optional: write the palette into the theme (needs the pptx skill's apply_theme.js).
  const themeJs = process.env.APPLY_THEME_JS;
  if (themeJs) {
    const { applyTheme } = require(themeJs);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
